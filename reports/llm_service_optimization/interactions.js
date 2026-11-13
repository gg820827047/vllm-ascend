"use strict";
(() => {
  const data = JSON.parse(document.getElementById("report-data").textContent);
  const $ = (id) => document.getElementById(id);
  const controls = ["category", "stage", "status", "role", "query"];
  const rows = new Map([...document.querySelectorAll("[data-task]")].map((r) => [r.dataset.task, r]));
  let visible = data.tasks;

  function applyFilters() {
    const q = $("query").value.trim().toLocaleLowerCase();
    visible = data.tasks.filter((t) =>
      (!$("category").value || t.category === $("category").value) &&
      (!$("stage").value || t.stages.includes($("stage").value) || t.stages.includes("X")) &&
      (!$("status").value || t.status === $("status").value) &&
      (!$("role").value || t.owner_role === $("role").value || t.collaborators.includes($("role").value)) &&
      (!q || JSON.stringify(t).toLocaleLowerCase().includes(q))
    );
    const ids = new Set(visible.map((t) => t.id));
    rows.forEach((row, id) => { row.hidden = !ids.has(id); });
    $("result-count").textContent = `${visible.length} / ${data.tasks.length} 项`;
    $("empty").hidden = visible.length !== 0;
    document.querySelectorAll("[data-stage]").forEach((button) => {
      button.setAttribute("aria-pressed", String(button.dataset.stage === $("stage").value));
    });
    const stage = data.stages.find((s) => s.id === $("stage").value);
    $("stage-context").hidden = !stage;
    if (stage) {
      $("stage-heading").textContent = `${stage.id} · ${stage.title}`;
      $("stage-meaning").textContent = stage.meaning;
      $("stage-code").textContent = `代码边界：${stage.code}`;
      $("stage-observe").textContent = `观测：${stage.observe}`;
    }
  }

  function reset() {
    controls.forEach((id) => { $(id).value = ""; });
    applyFilters();
  }

  function quoteCSV(value) {
    return `"${String(value ?? "").replaceAll('"', '""')}"`;
  }

  function exportCSV() {
    const fields = [
      ["id", "ID"], ["category_name", "主题"], ["stage_names", "请求位置"], ["title", "子事项"],
      ["kind", "任务类型"], ["priority", "优先级"], ["status", "当前状态"], ["owner", "责任人"],
      ["role_name", "建议责任角色"], ["collaborator_names", "协同角色"], ["dependency_names", "前置任务"],
      ["current", "已有证据与范围"], ["action_text", "分析或实施步骤"], ["deliverable", "交付物"],
      ["expected_benefit", "预期收益"], ["metric", "验证指标"], ["acceptance", "验收条件"],
      ["evidence_names", "证据索引"],
    ];
    const lines = [fields.map(([, title]) => quoteCSV(title)).join(",")];
    visible.forEach((task) => lines.push(fields.map(([key]) => quoteCSV(task[key])).join(",")));
    const url = URL.createObjectURL(new Blob(["\uFEFF", lines.join("\r\n") + "\r\n"], {type: "text/csv;charset=utf-8"}));
    const link = document.createElement("a");
    link.href = url;
    link.download = `llm-optimization-${data.date}-${visible.length}-tasks.csv`;
    link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  controls.forEach((id) => $(id).addEventListener(id === "query" ? "input" : "change", applyFilters));
  $("reset").addEventListener("click", reset);
  $("export").addEventListener("click", exportCSV);
  document.querySelectorAll("[data-stage]").forEach((button) => button.addEventListener("click", () => {
    $("stage").value = $("stage").value === button.dataset.stage ? "" : button.dataset.stage;
    applyFilters();
  }));
  document.querySelectorAll("[data-category]").forEach((button) => button.addEventListener("click", () => {
    reset();
    $("category").value = button.dataset.category;
    applyFilters();
    $("tasks").scrollIntoView({behavior: "smooth", block: "start"});
  }));
  $("expand").addEventListener("click", () => {
    visible.forEach((task) => { rows.get(task.id).querySelector("details").open = true; });
  });
  $("collapse").addEventListener("click", () => {
    document.querySelectorAll(".task-detail details").forEach((item) => { item.open = false; });
  });
  document.addEventListener("click", (event) => {
    const link = event.target.closest('a[href^="#task-"]');
    if (!link) return;
    const row = document.getElementById(link.getAttribute("href").slice(1));
    if (!row) return;
    reset();
    row.querySelector("details").open = true;
  });
  applyFilters();
})();
