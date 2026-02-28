import server
import json
import urllib.request
import urllib.parse
import traceback
import torch
import comfy.model_management

class AnyType(str):
    """A special type that can connect to any input/output in ComfyUI."""
    def __ne__(self, __value: object) -> bool:
        return False

# Wildcard type to accept IMAGE, LATENT, STRING, etc.
ANY_TYPE = AnyType("*")

# Global Cache to store large data (like Tensors) across independent loop iterations.
# ComfyUI's JSON prompt cannot serialize Python Tensor objects directly.
# Key: node_id (of LoopStart), Value: dict{"current_iteration": INT, "data": ANY}
LOOP_STATE = {}

class MidnightLook_LoopStart:
    """
    Acts as the entry point for the loop.
    Initializes or tracks the current iteration by checking the global LOOP_STATE cache.
    Outputs the payload data, current iteration, and max iterations for downstream nodes.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "initial_data_in": (ANY_TYPE,),
                "max_iterations": ("INT", {"default": 10, "min": 1, "max": 100}),
            },
            "optional": {
                "_force_rerun": ("INT", {"default": 0}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            }
        }

    RETURN_TYPES = (ANY_TYPE, "INT", "INT")
    RETURN_NAMES = ("loop_data_out", "current_iteration", "max_iterations")
    FUNCTION = "start_loop"
    CATEGORY = "Midnight Look/Loop"

    def start_loop(self, initial_data_in, max_iterations, unique_id=None, **kwargs):
        global LOOP_STATE
        
        iteration = 1
        output = initial_data_in
        
        # Check if we are currently looping this specific node
        if unique_id in LOOP_STATE:
            state = LOOP_STATE[unique_id]
            iteration = state.get("current_iteration", 1)
            output = state.get("data", initial_data_in)
            
            # Safeguard: if we exceed max_iterations (e.g. user manually reduced max_iterations mid-loop), reset
            if iteration > max_iterations:
                print(f"[Midnight Look Loop] Resetting loop at node {unique_id} due to max iterations safeguard.")
                del LOOP_STATE[unique_id]
                iteration = 1
                output = initial_data_in
                
        return (output, iteration, max_iterations)

    @classmethod
    def IS_CHANGED(s, **kwargs):
        # Force ComfyUI to always re-evaluate this node so the loop iteration counter increases.
        return float("NaN")


class MidnightLook_LoopEnd:
    """
    End Loop & Decision Maker.
    If condition == True or current >= max: Releases final data and completes the loop.
    If condition == False: Halts downstream execution, saves data to cache, and re-queues the workflow API.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "processed_data_in": (ANY_TYPE,),
                "condition": ("BOOLEAN", {"default": False}),
                "current_iteration": ("INT", {"default": 1, "forceInput": True}),
                "max_iterations": ("INT", {"default": 10, "forceInput": True}),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
                "unique_id": "UNIQUE_ID",
            }
        }

    RETURN_TYPES = (ANY_TYPE,)
    RETURN_NAMES = ("final_data_out",)
    FUNCTION = "end_loop"
    CATEGORY = "Midnight Look/Loop"
    OUTPUT_NODE = True 

    def end_loop(self, processed_data_in, condition, current_iteration, max_iterations, prompt=None, extra_pnginfo=None, unique_id=None):
        global LOOP_STATE
        
        # Find the Loop Start Node ID dynamically by tracing the 'current_iteration' input link.
        # This completely bypasses the need for the user to manually type the Node ID.
        loop_start_id = None
        if prompt and unique_id in prompt:
            inputs = prompt[unique_id].get("inputs", {})
            # 'current_iteration' is linked from Loop Start. It arrives as ["NodeID", OutputIndex]
            if "current_iteration" in inputs and isinstance(inputs["current_iteration"], list):
                loop_start_id = inputs["current_iteration"][0]

        # Escape condition: Success OR Safety Limit reached
        if condition or current_iteration >= max_iterations:
            print(f"[Midnight Look Loop] ✅ Loop completed at iteration {current_iteration}. condition={condition}. Advancing workflow.")
            
            # Clean up the cache to prevent ghost data on next fresh run
            if loop_start_id and loop_start_id in LOOP_STATE:
                del LOOP_STATE[loop_start_id]
                
            # Returning allows all downstream nodes connected to 'final_data_out' to execute normally
            return (processed_data_in,)
            
        print(f"[Midnight Look Loop] 🔄 Iteration {current_iteration}/{max_iterations} condition not met. Re-queuing loop...")
        
        if loop_start_id:
            # 1. Store processed data in memory cache for the next run
            LOOP_STATE[loop_start_id] = {
                "current_iteration": current_iteration + 1,
                "data": processed_data_in
            }
            
            # 2. Re-queue the prompt to trigger the new iteration
            if prompt:
                import copy
                import random
                
                # Deepcopy to safely modify the prompt graph for the new run
                new_prompt = copy.deepcopy(prompt)
                
                # Randomize seeds in standard nodes to ensure new generation and bypass cache
                for node_id, node_info in new_prompt.items():
                    if "inputs" in node_info:
                        inputs = node_info["inputs"]
                        # Target common seed fields in KSamplers and other generative nodes
                        for seed_key in ["seed", "noise_seed"]:
                            if seed_key in inputs and isinstance(inputs[seed_key], (int, float)):
                                # Replace with a new random 64-bit integer
                                inputs[seed_key] = random.randint(0, 0xffffffffffffffff)

                # Force prompt uniqueness by altering Loop Start's input
                if loop_start_id in new_prompt:
                    if "inputs" not in new_prompt[loop_start_id]:
                        new_prompt[loop_start_id]["inputs"] = {}
                    new_prompt[loop_start_id]["inputs"]["_force_rerun"] = random.randint(0, 0xffffffff)

                p = {"prompt": new_prompt}
                # Include exact UI schema if available, useful for front-end history
                if extra_pnginfo:
                    p["extra_data"] = {"extra_pnginfo": extra_pnginfo}

                # Retrieve client_id so the UI shows progress bar and green borders for the new run
                if hasattr(server.PromptServer.instance, "client_id"):
                    p["client_id"] = server.PromptServer.instance.client_id
                    
                data = json.dumps(p).encode('utf-8')
                req = urllib.request.Request("http://127.0.0.1:8188/prompt", data=data)
                req.add_header('Content-Type', 'application/json')
                
                try:
                    urllib.request.urlopen(req)
                    print(f"[Midnight Look Loop] Successfully queued iteration {current_iteration + 1} to API.")
                except Exception as e:
                    print(f"[Midnight Look Loop] ❌ Error re-queuing prompt to API: {e}")
                    traceback.print_exc()

        # 3. Halt downstream execution for this current iteration (Lazy Evaluation)
        print(f"[Midnight Look Loop] Halting downstream execution for iteration {current_iteration}.")
        comfy.model_management.interrupt_current_processing(True)
        
        # Return something to satisfy signature; downstream ignores this because they were interrupted
        return (processed_data_in,)

NODE_CLASS_MAPPINGS = {
    "MidnightLook_LoopStart": MidnightLook_LoopStart,
    "MidnightLook_LoopEnd": MidnightLook_LoopEnd,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MidnightLook_LoopStart": "Loop Start (Midnight Look)",
    "MidnightLook_LoopEnd": "Loop End / Condition (Midnight Look)",
}
