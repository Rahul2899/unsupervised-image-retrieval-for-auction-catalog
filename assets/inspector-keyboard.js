(function () {
  document.addEventListener("keydown", function (event) {
    if (!document.querySelector("#inspector.is-open")) return;
    if (!["Escape", "ArrowLeft", "ArrowRight"].includes(event.key)) return;
    event.preventDefault();
    if (window.dash_clientside && typeof window.dash_clientside.set_props === "function") {
      window.dash_clientside.set_props("inspector-key", {
        data: { key: event.key, stamp: Date.now() }
      });
    }
  });
})();
