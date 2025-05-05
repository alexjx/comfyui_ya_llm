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
    // handle of OllamaChat and OllamaGenerate nodes
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
  },
   
});
