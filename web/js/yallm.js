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
    if (["yaOllamaChat", "yaOllamaGenerate",].includes(nodeType?.prototype.comfyClass)) {
      const originalNodeCreated = nodeType.prototype.onNodeCreated;

      nodeType.prototype.onNodeCreated = async function () {
        if (originalNodeCreated) {
          originalNodeCreated.apply(this, arguments);
        }

        const urlWidget = this.widgets.find((w) => w.name === "url");
        const modelWidget = this.widgets.find((w) => w.name === "model");

        const updateModels = async () => {
          const url = urlWidget.value;
          const prevValue = modelWidget.value
          modelWidget.value = ''
          modelWidget.options.values = []

          const models = await fetchModels(url);

          // Update modelWidget options and value
          modelWidget.options.values = models;
          console.debug("Updated modelWidget.options.values:", modelWidget.options.values);

          if (models.includes(prevValue)) {
            modelWidget.value = prevValue; // stay on current.
          } else if (models.length > 0) {
            modelWidget.value = models[0]; // set first as default.
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

    // Handle TextTemplate node - dynamic inputs
    if (nodeType?.prototype.comfyClass === "yaLLMTextTemplate") {
      const TypeSlot = { Input: 1, Output: 2 };
      const TypeSlotEvent = { Connect: true, Disconnect: false };
      const PREFIX = "input";

      const onNodeCreated = nodeType.prototype.onNodeCreated;
      nodeType.prototype.onNodeCreated = async function () {
        const me = onNodeCreated?.apply(this);

        // Initialize stable counter for this node instance
        if (!this.inputCounter) {
          this.inputCounter = 0;
        }

        // Start with input_1
        this.inputCounter++;
        this.addInput(`${PREFIX}_${this.inputCounter}`, "*");
        const slot = this.inputs[this.inputs.length - 1];
        if (slot) {
          slot.color_off = "#666";
        }

        return me;
      };

      const onConnectionsChange = nodeType.prototype.onConnectionsChange;
      nodeType.prototype.onConnectionsChange = function (slotType, slot_idx, event, link_info, node_slot) {
        const me = onConnectionsChange?.apply(this, arguments);

        if (slotType === TypeSlot.Input) {
          if (event === TypeSlotEvent.Connect && link_info) {
            // Set the type based on connected node
            const fromNode = this.graph._nodes.find(
              (otherNode) => otherNode.id == link_info.origin_id
            );
            if (fromNode) {
              const parent_link = fromNode.outputs[link_info.origin_slot];
              if (parent_link) {
                node_slot.type = parent_link.type;
              }
            }
          } else if (event === TypeSlotEvent.Disconnect) {
            // When disconnecting, just remove the input - don't renumber
            this.removeInput(slot_idx);
          }

          // Remove any unconnected inputs (except keep one empty slot at the end)
          for (let i = this.inputs.length - 1; i >= 0; i--) {
            const slot = this.inputs[i];
            if (slot.link === null) {
              // Keep one unconnected slot
              const unconnectedCount = this.inputs.filter(s => s.link === null).length;
              if (unconnectedCount > 1) {
                try {
                  this.removeInput(i);
                } catch (e) {
                  // Ignore errors when removing
                }
              }
            }
          }

          // Ensure there's always one empty slot at the end for the next connection
          const hasUnconnected = this.inputs.some(s => s.link === null);
          if (!hasUnconnected) {
            // Initialize counter if needed (for loaded workflows)
            if (!this.inputCounter) {
              // Find the highest existing input number
              let maxNum = 0;
              for (const input of this.inputs) {
                const match = input.name.match(/input_(\d+)/);
                if (match) {
                  maxNum = Math.max(maxNum, parseInt(match[1]));
                }
              }
              this.inputCounter = maxNum;
            }

            this.inputCounter++;
            this.addInput(`${PREFIX}_${this.inputCounter}`, "*");
            const newSlot = this.inputs[this.inputs.length - 1];
            if (newSlot) {
              newSlot.color_off = "#666";
            }
          }

          // Force the node to resize itself for the new/deleted connections
          this?.graph?.setDirtyCanvas(true);
        }

        return me;
      };
    }
  },
   
});
