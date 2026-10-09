import { app } from "/scripts/app.js";

const stylesheet = document.createElement("link");
stylesheet.rel = "stylesheet";
stylesheet.href = new URL("./template_replace.css", import.meta.url).href;
document.head.appendChild(stylesheet);

function installReplacePanel(node, widget) {
  const textarea = widget.element;
  let overlay;
  let controller;
  let originalPadding;

  const cleanup = () => {
    controller?.abort();
    overlay?.remove();
    if (originalPadding !== undefined) textarea.style.paddingRight = originalPadding;
    overlay = null;
    controller = null;
  };

  const writeValue = (value) => {
    const canvas = node.graph === app.canvas?.graph ? app.canvas : null;
    if (canvas) {
      app.extensionManager.workflow.activeWorkflow?.changeTracker?.captureCanvasState();
      canvas.emitBeforeChange();
    }
    try {
      widget.value = value;
      node.graph?.setDirtyCanvas(true, true);
    } finally {
      canvas?.emitAfterChange();
    }
  };

  const mount = () => {
    const host = textarea.parentElement;
    if (!textarea.isConnected || !host?.classList.contains("dom-widget")) return;
    if (overlay?.parentElement === host) return;
    cleanup();
    controller = new AbortController();
    const { signal } = controller;
    let lastReplacement;

    overlay = document.createElement("div");
    overlay.className = "yallm-template-replace";
    overlay.innerHTML = `
      <button type="button" class="yallm-replace-toggle" title="批量替换" aria-expanded="false">批量替换</button>
      <form class="yallm-replace-panel" aria-label="批量替换" hidden>
        <div class="yallm-replace-header">
          <strong>批量替换</strong>
          <button type="button" class="yallm-replace-close">关闭</button>
        </div>
        <label><span>查找内容</span><textarea class="yallm-replace-search" rows="2" placeholder="输入要替换的文本" spellcheck="false"></textarea></label>
        <label><span>替换为</span><textarea class="yallm-replace-value" rows="2" placeholder="留空表示删除" spellcheck="false"></textarea></label>
        <div class="yallm-replace-actions">
          <button type="submit" class="yallm-replace-apply" disabled>替换全部</button>
          <button type="button" class="yallm-replace-undo" hidden>撤销替换</button>
        </div>
        <output class="yallm-replace-status" role="status" aria-live="polite"></output>
      </form>`;

    const toggle = overlay.querySelector(".yallm-replace-toggle");
    const panel = overlay.querySelector("form");
    const search = overlay.querySelector(".yallm-replace-search");
    const replacement = overlay.querySelector(".yallm-replace-value");
    const apply = overlay.querySelector(".yallm-replace-apply");
    const undo = overlay.querySelector(".yallm-replace-undo");
    const status = overlay.querySelector("output");

    const update = (message) => {
      const text = widget.value;
      const query = search.value;
      let count = 0;
      if (query) {
        for (let start = 0; (start = text.indexOf(query, start)) !== -1; start += query.length) count++;
      }
      apply.disabled = count === 0;
      undo.hidden = !lastReplacement;
      undo.disabled = !lastReplacement || text !== lastReplacement.after;
      status.textContent = message || (query ? `匹配 ${count} 处（区分大小写）` : "请输入查找内容");
      return count;
    };

    const close = () => {
      panel.hidden = true;
      toggle.hidden = false;
      toggle.setAttribute("aria-expanded", "false");
      toggle.focus();
    };

    toggle.addEventListener("click", () => {
      panel.hidden = false;
      toggle.hidden = true;
      toggle.setAttribute("aria-expanded", "true");
      update();
      search.focus();
    }, { signal });
    overlay.querySelector(".yallm-replace-close").addEventListener("click", close, { signal });
    search.addEventListener("input", () => update(), { signal });
    replacement.addEventListener("input", () => update(), { signal });
    textarea.addEventListener("input", () => {
      lastReplacement = null;
      update();
    }, { signal });

    panel.addEventListener("submit", (event) => {
      event.preventDefault();
      const count = update();
      if (!count) return;
      const before = widget.value;
      const after = before.replaceAll(search.value, () => replacement.value);
      if (after === before) {
        update("内容未改变");
        return;
      }
      writeValue(after);
      lastReplacement = { before, after };
      update(`已替换 ${count} 处`);
    }, { signal });
    undo.addEventListener("click", () => {
      if (!lastReplacement || widget.value !== lastReplacement.after) return;
      writeValue(lastReplacement.before);
      lastReplacement = null;
      update("已撤销替换");
    }, { signal });

    for (const type of ["pointerdown", "pointermove", "pointerup", "mousedown", "mouseup", "click", "wheel"]) {
      overlay.addEventListener(type, (event) => event.stopPropagation(), { signal });
    }
    overlay.addEventListener("keydown", (event) => {
      event.stopPropagation();
      if (event.key === "Escape") {
        event.preventDefault();
        close();
      }
    }, { signal });

    originalPadding = textarea.style.paddingRight;
    textarea.style.paddingRight = "96px";
    host.appendChild(overlay);
    update();
  };

  const onDraw = widget.options.onDraw;
  widget.options.onDraw = function () {
    const result = onDraw?.apply(this, arguments);
    mount();
    return result;
  };
  const onRemove = widget.onRemove;
  widget.onRemove = function () {
    cleanup();
    return onRemove?.apply(this, arguments);
  };
}

app.registerExtension({
  name: "Comfy.yallm.TemplateReplace",
  nodeCreated(node) {
    if (node.comfyClass !== "yaLLMTextTemplate") return;
    const widget = node.widgets?.find((widget) => widget.name === "template");
    if (widget?.element instanceof HTMLTextAreaElement) installReplacePanel(node, widget);
  },
});
