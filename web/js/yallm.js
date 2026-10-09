import { app } from "/scripts/app.js";

const fetchModels = async (url) => {
  try {
    const response = await fetch("/yallm/get_ollama_models", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        url,
      }),
    });

    if (response.ok) {
      const models = await response.json();
      console.debug("yallmNodes: Fetched models:", models);
      return models;
    } else {
      console.error(`yallmNodes: Failed to fetch models: ${response.status}`);
      return [];
    }
  } catch (error) {
    console.error(`yallmNodes: Error fetching models`, error);
    return [];
  }
};

// array of registered nodes
const registeredNodes = {};

app.registerExtension({
  name: "Comfy.yallmNode",
  
  async beforeRegisterNodeDef(nodeType, nodeData, app) {
    // Handle OllamaChat and OllamaGenerate nodes
    if (["yaOllamaChat", "yaOllamaGenerate", "yaSEGSCaptioner", "yaSEGSCaptionerV2"].includes(nodeType?.prototype.comfyClass)) {
      const originalNodeCreated = nodeType.prototype.onNodeCreated;

      nodeType.prototype.onNodeCreated = async function () {
        if (originalNodeCreated) {
          originalNodeCreated.apply(this, arguments);
        }

        const urlWidget = this.widgets.find((w) => w.name === "url");
        const modelWidget = this.widgets.find((w) => w.name === "model");

        const updateModels = async () => {
          const url = urlWidget.value;
          const prevValue = modelWidget.value;

          const models = await fetchModels(url);

          // Update modelWidget options
          modelWidget.options.values = models;
          console.debug("Updated modelWidget.options.values:", modelWidget.options.values);

          // Only change the value if the previous model is not in the new list
          if (models.includes(prevValue)) {
            modelWidget.value = prevValue; // Keep the current model
          } else if (models.length > 0) {
            modelWidget.value = models[0]; // Set first as default only if current model doesn't exist
          } else {
            modelWidget.value = ''; // Clear if no models available
          }

          console.debug("Updated modelWidget.value:", modelWidget.value);
        };

        urlWidget.callback = updateModels;

        const dummy = async () => {
          // calling async method will update the widgets with actual value from the browser and not the default from Node definition.
        }

        // Initial update
        await dummy(); // this will cause the widgets to obtain the actual value from web page.
        await updateModels();

        // save the updateModels method for refreshing the models later.
        if (!registeredNodes[nodeType.comfyClass]) {
          registeredNodes[nodeType.comfyClass] = [];
        }
        registeredNodes[nodeType.comfyClass].push(updateModels);
      };

      // refresh the existing nodes
      if (registeredNodes[nodeType.comfyClass]) {
        for (const updateModels of registeredNodes[nodeType.comfyClass]) {
          await updateModels();
        }
      }
    }

    // Handle TextTemplate and ImageLister nodes - dynamic inputs
    if (["yaLLMTextTemplate", "yaImageLister"].includes(nodeType?.prototype.comfyClass)) {
      const PREFIX = nodeType.prototype.comfyClass === "yaImageLister" ? "image" : "input";
      const inputNamePattern = new RegExp(`^${PREFIX}_\\d+$`);
      const isDynamicInput = (slot) => !slot.widget && inputNamePattern.test(slot.name);
      const configuringInputs = Symbol("yallmConfiguringInputs");
      const updatingInputs = Symbol("yallmUpdatingInputs");
      const restoredInputNames = Symbol("yallmRestoredInputNames");

      const updateInputs = (node) => {
        if (node[updatingInputs]) return;
        node[updatingInputs] = true;
        try {
          const dynamicInputs = node.inputs.filter(isDynamicInput);
          const lastInput = dynamicInputs[dynamicInputs.length - 1];
          const emptyInput = lastInput && !node.isInputConnected(node.inputs.indexOf(lastInput)) ? lastInput : null;

          // Remove only empty dynamic sockets, leaving one at the end.
          for (let i = node.inputs.length - 1; i >= 0; i--) {
            const slot = node.inputs[i];
            if (isDynamicInput(slot) && !node.isInputConnected(i) && slot !== emptyInput) {
              node.removeInput(i);
            }
          }

          let count = 0;
          for (let i = 0; i < node.inputs.length; i++) {
            const slot = node.inputs[i];
            if (!isDynamicInput(slot)) continue;
            slot.name = `${PREFIX}_${++count}`;
            const type = node.graph ? node.getInputDataType(i) : null;
            if (type != null) slot.type = type;
          }

          const spare = emptyInput || node.addInput(`${PREFIX}_${count + 1}`, "*");
          spare.type = "*";
          spare.color_off = "#666";
          node.graph?.setDirtyCanvas(true);
        } finally {
          node[updatingInputs] = false;
        }
      };

      const onNodeCreated = nodeType.prototype.onNodeCreated;
      nodeType.prototype.onNodeCreated = function () {
        const me = onNodeCreated?.apply(this, arguments);
        updateInputs(this);
        return me;
      };

      const configure = nodeType.prototype.configure;
      nodeType.prototype.configure = function (info) {
        // Old clipboard data uses socket indices and may omit the fixed template socket.
        this[restoredInputNames] = PREFIX === "input" && !app.configuringGraph && info.inputs &&
          !info.inputs.some((slot) => slot.name === "template") &&
          info.inputs.every((slot) => slot.link == null)
          ? info.inputs.map((slot) => slot.name)
          : null;

        // Older workflows can have dynamic sockets incorrectly bound to template.
        if (PREFIX === "input" && info.inputs) {
          info.inputs = info.inputs.map((slot) => {
            if (inputNamePattern.test(slot.name) && slot.widget?.name === "template") {
              const input = { ...slot };
              delete input.widget;
              delete input.localized_name;
              return input;
            }
            return slot;
          });
        }

        this[configuringInputs] = true;
        try {
          return configure.apply(this, [info]);
        } finally {
          if (app.configuringGraph) {
            this[configuringInputs] = false;
            this[restoredInputNames] = null;
          } else {
            // Clipboard paste reconnects saved socket indices after configure returns.
            queueMicrotask(() => {
              this[configuringInputs] = false;
              this[restoredInputNames] = null;
              updateInputs(this);
            });
          }
        }
      };

      const onBeforeConnectInput = nodeType.prototype.onBeforeConnectInput;
      nodeType.prototype.onBeforeConnectInput = function (slot_idx, target_slot) {
        const names = this[restoredInputNames];
        if (this[configuringInputs] && names && typeof target_slot === "number") {
          const restoredSlot = this.findInputSlot(names[slot_idx]);
          if (restoredSlot >= 0) slot_idx = restoredSlot;
        }
        return onBeforeConnectInput ? onBeforeConnectInput.call(this, slot_idx, target_slot) : slot_idx;
      };

      const onGraphConfigured = nodeType.prototype.onGraphConfigured;
      nodeType.prototype.onGraphConfigured = function () {
        const me = onGraphConfigured?.apply(this, arguments);
        updateInputs(this);
        return me;
      };

      const onConnectionsChange = nodeType.prototype.onConnectionsChange;
      nodeType.prototype.onConnectionsChange = function (slotType, slot_idx, event, link_info, node_slot) {
        const me = onConnectionsChange?.apply(this, arguments);

        // Restoring slots and links is incomplete until onGraphConfigured.
        if (
          slotType !== 1 || app.configuringGraph || this[configuringInputs] ||
          this[updatingInputs] || !node_slot || !isDynamicInput(node_slot)
        ) {
          return me;
        }

        // A replacement connection may already occupy a disconnected socket.
        updateInputs(this);
        return me;
      };
    }
  },
   
});
