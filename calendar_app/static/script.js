document.addEventListener("DOMContentLoaded", () => {
  const calendarEl = document.getElementById("calendar");
  const form = document.getElementById("event-form");
  const startInput = document.getElementById("start");
  const endInput = document.getElementById("end");
  const fleetInput = document.getElementById("fleet");
  const doctrineInput = document.getElementById("doctrine");
  const handoutInput = document.getElementById("handout");
  const addEventButton = document.getElementById("add-event");
  const statusEl = document.getElementById("status");
  const summaryCard = document.getElementById("summary-card");
  const summaryBody = summaryCard.querySelector(".summary-body");

  let selection = null;

  const calendar = new FullCalendar.Calendar(calendarEl, {
    initialView: "timeGridWeek",
    locale: "ko",
    slotMinTime: "06:00:00",
    slotMaxTime: "24:00:00",
    selectable: true,
    selectMirror: true,
    headerToolbar: {
      left: "prev,next today",
      center: "title",
      right: "dayGridMonth,timeGridWeek,timeGridDay,listWeek"
    },
    select(info) {
      selection = info;
      startInput.value = formatDateTime(info.start);
      endInput.value = formatDateTime(info.end);
      addEventButton.disabled = false;
      statusEl.textContent = "세부 정보를 입력한 후 일정을 추가하세요.";
    },
    eventClick(info) {
      info.jsEvent.preventDefault();
      const event = info.event;
      renderSummary({
        title: event.title,
        start: event.start,
        end: event.end,
        extendedProps: event.extendedProps
      });
    }
  });

  calendar.render();

  form.addEventListener("submit", (event) => {
    event.preventDefault();
    if (!selection) {
      statusEl.textContent = "캘린더에서 날짜와 시간을 먼저 선택해주세요.";
      return;
    }

    const eventData = {
      title: `${fleetInput.value.trim()} | ${doctrineInput.value.trim()}`,
      start: selection.start,
      end: selection.end,
      extendedProps: {
        fleet: fleetInput.value.trim(),
        doctrine: doctrineInput.value.trim(),
        handout: handoutInput.value
      }
    };

    calendar.addEvent(eventData);
    calendar.unselect();
    selection = null;
    addEventButton.disabled = true;

    renderSummary(eventData);
    statusEl.textContent = "일정이 추가되었습니다. 오른쪽 요약을 확인하세요.";

    form.reset();
  });

  function formatDateTime(date) {
    return new Intl.DateTimeFormat("ko-KR", {
      dateStyle: "medium",
      timeStyle: "short"
    }).format(date);
  }

  function renderSummary(eventData) {
    const { start, end, extendedProps } = eventData;
    const summaryTemplate = `
      <dl>
        <div>
          <dt>시작</dt>
          <dd>${formatDateTime(start)}</dd>
        </div>
        <div>
          <dt>종료</dt>
          <dd>${formatDateTime(end)}</dd>
        </div>
        <div>
          <dt>플릿 컴포</dt>
          <dd>${escapeHtml(extendedProps?.fleet ?? "")}</dd>
        </div>
        <div>
          <dt>독트린</dt>
          <dd>${escapeHtml(extendedProps?.doctrine ?? "")}</dd>
        </div>
        <div>
          <dt>핸드아웃 여부</dt>
          <dd>${escapeHtml(extendedProps?.handout ?? "")}</dd>
        </div>
      </dl>
    `;

    summaryBody.innerHTML = summaryTemplate;
  }

  function escapeHtml(value) {
    return value
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }
});
