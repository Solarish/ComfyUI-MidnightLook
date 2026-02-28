import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const EXTENSION_NAME = "MidnightLook.ImageCompare";

app.registerExtension({
    name: EXTENSION_NAME,
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "MidnightLook_ImageCompare") {

            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

                this.ml_ui_widget = null;
                this.serialize_widgets = true;

                // Create a dominant HTML element to host our UI
                const container = document.createElement("div");
                container.style.position = "relative";
                container.style.width = "100%";
                container.style.height = "100%";
                container.style.pointerEvents = "auto";
                container.style.overflow = "hidden";
                container.style.borderRadius = "8px";
                container.style.backgroundColor = "#222";

                // Add DOM widget
                this.ml_ui_widget = this.addDOMWidget("compare_view", "div", container, {
                    serialize: false,
                    hideOnZoom: false
                });

                // Style the widget properly in LiteGraph
                this.ml_ui_widget.computeSize = function (width) {
                    return [width, this.last_h || 250];
                };

                // Elements
                this.ml_img1 = document.createElement("img");
                this.ml_img2 = document.createElement("img");
                this.ml_slider = document.createElement("input");
                this.ml_score = document.createElement("div");

                // Setup Image 1 (Bottom / Before)
                this.ml_img1.style.position = "absolute";
                this.ml_img1.style.top = "0";
                this.ml_img1.style.left = "0";
                this.ml_img1.style.width = "100%";
                this.ml_img1.style.height = "100%";
                this.ml_img1.style.objectFit = "contain";

                // Setup Image 2 (Top / After)
                this.ml_img2.style.position = "absolute";
                this.ml_img2.style.top = "0";
                this.ml_img2.style.left = "0";
                this.ml_img2.style.width = "100%";
                this.ml_img2.style.height = "100%";
                this.ml_img2.style.objectFit = "contain";
                this.ml_img2.style.clipPath = "polygon(0 0, 50% 0, 50% 100%, 0 100%)";

                // Setup Slider
                this.ml_slider.type = "range";
                this.ml_slider.min = "0";
                this.ml_slider.max = "100";
                this.ml_slider.value = "50";
                this.ml_slider.style.position = "absolute";
                this.ml_slider.style.top = "0";
                this.ml_slider.style.left = "0";
                this.ml_slider.style.width = "100%";
                this.ml_slider.style.height = "100%";
                this.ml_slider.style.opacity = "0";
                this.ml_slider.style.cursor = "col-resize";
                this.ml_slider.style.zIndex = "10";
                this.ml_slider.className = "midnight-slider";

                // Draw visible line via pseudo element or simple div (since range is invisible)
                this.ml_line = document.createElement("div");
                this.ml_line.style.position = "absolute";
                this.ml_line.style.top = "0";
                this.ml_line.style.left = "50%";
                this.ml_line.style.width = "2px";
                this.ml_line.style.height = "100%";
                this.ml_line.style.backgroundColor = "white";
                this.ml_line.style.boxShadow = "0 0 4px rgba(0,0,0,0.5)";
                this.ml_line.style.pointerEvents = "none";
                this.ml_line.style.zIndex = "5";

                this.ml_handle = document.createElement("div");
                this.ml_handle.style.position = "absolute";
                this.ml_handle.style.top = "50%";
                this.ml_handle.style.left = "50%";
                this.ml_handle.style.transform = "translate(-50%, -50%)";
                this.ml_handle.style.width = "20px";
                this.ml_handle.style.height = "20px";
                this.ml_handle.style.borderRadius = "50%";
                this.ml_handle.style.backgroundColor = "white";
                this.ml_handle.style.boxShadow = "0 0 4px rgba(0,0,0,0.5)";
                this.ml_handle.style.pointerEvents = "none";
                this.ml_handle.style.zIndex = "6";

                // Setup Score
                this.ml_score.style.position = "absolute";
                this.ml_score.style.bottom = "8px";
                this.ml_score.style.left = "50%";
                this.ml_score.style.transform = "translateX(-50%)";
                this.ml_score.style.background = "rgba(0,0,0,0.7)";
                this.ml_score.style.color = "white";
                this.ml_score.style.padding = "4px 8px";
                this.ml_score.style.borderRadius = "4px";
                this.ml_score.style.fontSize = "12px";
                this.ml_score.style.fontFamily = "sans-serif";
                this.ml_score.style.zIndex = "20";
                this.ml_score.style.pointerEvents = "none";
                this.ml_score.innerText = "Waiting for images...";

                // Interaction
                this.ml_slider.addEventListener("input", (e) => {
                    const val = e.target.value;
                    this.ml_img2.style.clipPath = `polygon(0 0, ${val}% 0, ${val}% 100%, 0 100%)`;
                    this.ml_line.style.left = `${val}%`;
                    this.ml_handle.style.left = `${val}%`;
                });

                // Stop node from dragging when using slider
                this.ml_slider.addEventListener("mousedown", (e) => {
                    e.stopPropagation();
                });

                container.appendChild(this.ml_img1);
                container.appendChild(this.ml_img2);
                container.appendChild(this.ml_line);
                container.appendChild(this.ml_handle);
                container.appendChild(this.ml_score);
                container.appendChild(this.ml_slider);

                this.size = [400, 300];
                return r;
            };

            const onExecuted = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = function (message) {
                onExecuted?.apply(this, arguments);
                if (message && message.bimgs) {
                    this._ml_update_images(message);
                }
            };

            if (!app.ml_image_compare_hooked) {
                app.ml_image_compare_hooked = true;
                api.addEventListener("executed", (e) => {
                    if (e.detail && e.detail.node && e.detail.output) {
                        const node = app.graph.getNodeById(e.detail.node);
                        if (node && node.type === "MidnightLook_ImageCompare" && node._ml_update_images) {
                            node._ml_update_images(e.detail.output);
                        }
                    }
                });
            }

            nodeType.prototype._ml_update_images = function (ui_data) {
                if (ui_data && ui_data.bimgs && ui_data.bimgs.length >= 2) {
                    const url1 = api.apiURL(`/view?filename=${encodeURIComponent(ui_data.bimgs[0].filename)}&type=${ui_data.bimgs[0].type || "temp"}&subfolder=${ui_data.bimgs[0].subfolder || ""}`);
                    const url2 = api.apiURL(`/view?filename=${encodeURIComponent(ui_data.bimgs[1].filename)}&type=${ui_data.bimgs[1].type || "temp"}&subfolder=${ui_data.bimgs[1].subfolder || ""}`);

                    this.ml_img1.src = url1;
                    this.ml_img2.src = url2;

                    if (ui_data.score) {
                        this.ml_score.innerText = ui_data.score[0];
                    }

                    // Attempt to auto-resize height based on first image aspect ratio once it loads
                    this.ml_img1.onload = () => {
                        if (this.ml_img1.naturalWidth > 0 && this.ml_img1.naturalHeight > 0) {
                            const aspect = this.ml_img1.naturalWidth / this.ml_img1.naturalHeight;
                            const targetH = this.size[0] / aspect;
                            if (this.ml_ui_widget) {
                                this.ml_ui_widget.last_h = targetH + 10;
                            }
                            this.size[1] = targetH + 40;
                            this.setDirtyCanvas(true, true);
                        }
                    };
                }
            };
        }
    }
});
