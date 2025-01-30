import { app } from "/scripts/app.js";

app.registerExtension({
  name: "Comfy.yallmNode",
  
  async beforeRegisterNodeDef(nodeType, nodeData, app) {
  
    if (["yaOllamaChat", "yaOllamaGenerate",].includes(nodeData.name)) {
      const originalNodeCreated = nodeType.prototype.onNodeCreated;
  
      nodeType.prototype.onNodeCreated = async function () {
        if (originalNodeCreated) {
          originalNodeCreated.apply(this, arguments);
        }

        const urlWidget = this.widgets.find((w) => w.name === "url");
        const modelWidget = this.widgets.find((w) => w.name === "model");

        const fetchModels = async (url) => {
          try {
            const response = await fetch("/ollama/get_models", {
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
              console.debug("Fetched models:", models);
              return models;
            } else {
              console.error(`Failed to fetch models: ${response.status}`);
              return [];
            }
          } catch (error) {
            console.error(`Error fetching models`, error);
            return [];
          }
        };

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
      };
    }
   
    if (["yaOllamaChatDual",].includes(nodeData.name)) {
      const originalNodeCreated = nodeType.prototype.onNodeCreated;
  
      nodeType.prototype.onNodeCreated = async function () {
        if (originalNodeCreated) {
          originalNodeCreated.apply(this, arguments);
        }

        const urlWidget = this.widgets.find((w) => w.name === "url");
        const model1Widget = this.widgets.find((w) => w.name === "model1");
        const model2Widget = this.widgets.find((w) => w.name === "model2");

        const fetchModels = async (url) => {
          try {
            const response = await fetch("/ollama/get_models", {
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
              console.debug("Fetched models:", models);
              return models;
            } else {
              console.error(`Failed to fetch models: ${response.status}`);
              return [];
            }
          } catch (error) {
            console.error(`Error fetching models`, error);
            return [];
          }
        };

        const updateModels = async () => {
          const url = urlWidget.value;
          const prevValue1 = model1Widget.value
          const prevValue2 = model2Widget.value
          model1Widget.value = ''
          model1Widget.options.values = []
          model2Widget.value = ''
          model2Widget.options.values = []
          
          const models = await fetchModels(url);

          // Update modelWidget options and value
          model1Widget.options.values = models;
          console.debug("Updated model1Widget.options.values:", model1Widget.options.values);
          model2Widget.options.values = models;
          console.debug("Updated model2Widget.options.values:", model2Widget.options.values);

          if (models.includes(prevValue1)) {
            model1Widget.value = prevValue1; // stay on current.
          } else if (models.length > 0) {
            model1Widget.value = models[0]; // set first as default.
          }
          console.debug("Updated model1Widget.value:", model1Widget.value);
          
          if (models.includes(prevValue2)) {
            model2Widget.value = prevValue2; // stay on current.
          } else if (models.length > 0) {
            model2Widget.value = models[0]; // set first as default.
          }
          console.debug("Updated model2Widget.value:", model2Widget.value);
        };

        urlWidget.callback = updateModels;

        const dummy = async () => {
          // calling async method will update the widgets with actual value from the browser and not the default from Node definition.
        }

        // Initial update
        await dummy(); // this will cause the widgets to obtain the actual value from web page.
        await updateModels();
      };
    }
  },
});
