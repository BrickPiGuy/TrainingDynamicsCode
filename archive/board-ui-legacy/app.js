const STORAGE_KEY = "kanban-calendar-board-v1";

const boardLists = {
  todo: document.getElementById("todo-list"),
  doing: document.getElementById("doing-list"),
  done: document.getElementById("done-list"),
};

const taskForm = document.getElementById("task-form");
const taskTitleInput = document.getElementById("task-title");
const taskStatusInput = document.getElementById("task-status");
const taskTemplate = document.getElementById("task-template");
const calendarGrid = document.getElementById("calendar-grid");
const calendarMonth = document.getElementById("calendar-month");
const seedDemoButton = document.getElementById("seed-demo");

const state = {
  draggedTaskId: null,
  monthCursor: startOfMonth(new Date()),
  tasks: loadTasks(),
};

function loadTasks() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    return raw ? JSON.parse(raw) : [];
  } catch {
    return [];
  }
}

function saveTasks() {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(state.tasks));
}

function createTask({ title, dueDate, status }) {
  return {
    id: crypto.randomUUID(),
    title: title.trim(),
    dueDate: dueDate || "",
    status,
  };
}

function formatDueDate(value) {
  if (!value) {
    return "No due date";
  }

  const [year, month, day] = value.split("-").map(Number);
  const date = new Date(year, month - 1, day);
  return new Intl.DateTimeFormat(undefined, {
    month: "short",
    day: "numeric",
    year: "numeric",
  }).format(date);
}

function startOfMonth(date) {
  return new Date(date.getFullYear(), date.getMonth(), 1);
}

function isSameDay(first, second) {
  return (
    first.getFullYear() === second.getFullYear() &&
    first.getMonth() === second.getMonth() &&
    first.getDate() === second.getDate()
  );
}

function formatLocalISO(date) {
  const year = date.getFullYear();
  const month = String(date.getMonth() + 1).padStart(2, "0");
  const day = String(date.getDate()).padStart(2, "0");
  return `${year}-${month}-${day}`;
}

function tasksForDate(date) {
  const key = formatLocalISO(date);
  return state.tasks.filter((task) => task.dueDate === key);
}

function updateCounts() {
  ["todo", "doing", "done"].forEach((status) => {
    document.querySelector(`[data-count-for="${status}"]`).textContent =
      state.tasks.filter((task) => task.status === status).length;
  });
}

function renderBoard() {
  Object.values(boardLists).forEach((list) => {
    list.replaceChildren();
  });

  state.tasks.forEach((task) => {
    const taskNode = taskTemplate.content.firstElementChild.cloneNode(true);
    taskNode.dataset.taskId = task.id;
    taskNode.querySelector(".task-title").textContent = task.title;
    taskNode.querySelector(".task-meta").textContent = formatDueDate(task.dueDate);

    taskNode.addEventListener("dragstart", () => {
      state.draggedTaskId = task.id;
      taskNode.classList.add("dragging");
    });

    taskNode.addEventListener("dragend", () => {
      state.draggedTaskId = null;
      taskNode.classList.remove("dragging");
      document
        .querySelectorAll(".column")
        .forEach((column) => column.classList.remove("drag-target"));
    });

    taskNode
      .querySelector(".delete-task")
      .addEventListener("click", () => deleteTask(task.id));

    boardLists[task.status].appendChild(taskNode);
  });

  updateCounts();
}

function renderCalendar() {
  calendarGrid.replaceChildren();

  const monthStart = startOfMonth(state.monthCursor);
  calendarMonth.textContent = new Intl.DateTimeFormat(undefined, {
    month: "long",
    year: "numeric",
  }).format(monthStart);

  const startOffset = monthStart.getDay();
  const gridStart = new Date(monthStart);
  gridStart.setDate(monthStart.getDate() - startOffset);

  const today = new Date();

  for (let index = 0; index < 42; index += 1) {
    const currentDate = new Date(gridStart);
    currentDate.setDate(gridStart.getDate() + index);

    const cell = document.createElement("article");
    cell.className = "calendar-cell";

    if (currentDate.getMonth() !== monthStart.getMonth()) {
      cell.classList.add("is-outside");
    }

    if (isSameDay(currentDate, today)) {
      cell.classList.add("is-today");
    }

    const datePill = document.createElement("div");
    datePill.className = "calendar-date";
    datePill.textContent = currentDate.getDate();
    cell.appendChild(datePill);

    tasksForDate(currentDate).forEach((task) => {
      const badge = document.createElement("span");
      badge.className = `calendar-task ${task.status}`;
      badge.textContent = task.title;
      cell.appendChild(badge);
    });

    calendarGrid.appendChild(cell);
  }
}

function rerender() {
  saveTasks();
  renderBoard();
  renderCalendar();
}

function addTask(event) {
  event.preventDefault();
  const formData = new FormData(taskForm);
  const title = String(formData.get("title") || "").trim();
  const dueDate = String(formData.get("dueDate") || "");
  const status = String(formData.get("status") || "todo");

  if (!title) {
    taskTitleInput.focus();
    return;
  }

  state.tasks.unshift(createTask({ title, dueDate, status }));
  taskForm.reset();
  taskStatusInput.value = "todo";
  rerender();
}

function deleteTask(taskId) {
  state.tasks = state.tasks.filter((task) => task.id !== taskId);
  rerender();
}

function moveTask(taskId, nextStatus) {
  state.tasks = state.tasks.map((task) =>
    task.id === taskId ? { ...task, status: nextStatus } : task
  );
  rerender();
}

function seedDemoTasks() {
  state.tasks = [
    createTask({
      title: "Outline sprint goals",
      dueDate: dateOffsetISO(0),
      status: "todo",
    }),
    createTask({
      title: "Design onboarding flow",
      dueDate: dateOffsetISO(2),
      status: "doing",
    }),
    createTask({
      title: "Publish weekly recap",
      dueDate: dateOffsetISO(5),
      status: "done",
    }),
  ];

  state.monthCursor = startOfMonth(new Date());
  rerender();
}

function dateOffsetISO(offset) {
  const date = new Date();
  date.setDate(date.getDate() + offset);
  return formatLocalISO(date);
}

taskForm.addEventListener("submit", addTask);
seedDemoButton.addEventListener("click", seedDemoTasks);

document.getElementById("prev-month").addEventListener("click", () => {
  state.monthCursor = new Date(
    state.monthCursor.getFullYear(),
    state.monthCursor.getMonth() - 1,
    1
  );
  renderCalendar();
});

document.getElementById("next-month").addEventListener("click", () => {
  state.monthCursor = new Date(
    state.monthCursor.getFullYear(),
    state.monthCursor.getMonth() + 1,
    1
  );
  renderCalendar();
});

document.querySelectorAll(".column").forEach((column) => {
  column.addEventListener("dragover", (event) => {
    event.preventDefault();
    column.classList.add("drag-target");
  });

  column.addEventListener("dragleave", () => {
    column.classList.remove("drag-target");
  });

  column.addEventListener("drop", (event) => {
    event.preventDefault();
    column.classList.remove("drag-target");

    if (!state.draggedTaskId) {
      return;
    }

    moveTask(state.draggedTaskId, column.dataset.status);
  });
});

rerender();
