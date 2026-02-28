import { app } from "../../scripts/app.js";
import { ComfyWidgets } from "../../scripts/widgets.js";

app.registerExtension({
    name: "MidnightLook.Score",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "MidnightScore" || nodeData.name === "MidnightLook_DisplayAny") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

                const widgetName = "text_display";
                const isMultiline = nodeData.name === "MidnightLook_DisplayAny";

                // Add a text widget to display the output
                const w = ComfyWidgets["STRING"](this, widgetName, ["STRING", { multiline: isMultiline }], app).widget;
                w.inputEl.readOnly = true;
                w.inputEl.style.opacity = 0.6;
                w.value = "Ready";
                // Prevent this widget from being saved in the workflow
                w.serialize = false;

                return r;
            };

            const onExecuted = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = function (message) {
                onExecuted?.apply(this, arguments);

                if (message && message.text && message.text.length > 0) {
                    const text = message.text[0];
                    const w = this.widgets?.find((w) => w.name === "text_display");
                    if (w) {
                        w.value = text;
                    }
                }
            };
        }
    },
});
