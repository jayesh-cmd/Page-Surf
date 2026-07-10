document.getElementById("askBtn").addEventListener("click", () => {
  const queryInput = document.getElementById("userQuery");
  const query = queryInput.value.trim();
  const askBtn = document.getElementById("askBtn");
  const responseBox = document.getElementById("responseBox");

  if (!query) return;

  // 1. Enter Loading State
  askBtn.disabled = true;
  askBtn.innerHTML = '<span class="loading"></span>Thinking...';
  responseBox.innerHTML = '<div style="color: #64748b; text-align: center; margin-top: 50px; font-style: italic;">Processing...</div>';

  chrome.tabs.query({ active: true, currentWindow: true }, function (tabs) {
    chrome.scripting.executeScript(
      {
        target: { tabId: tabs[0].id },
        function: getPageContent
      },
      async (injectionResults) => {
        try {
          if (!injectionResults || !injectionResults[0]) {
            throw new Error("Could not extract page content. Try reloading the tab.");
          }

          const pageContent = injectionResults[0].result;

          const response = await fetch("http://127.0.0.1:8000/chat", {
            method: "POST",
            headers: {
              "Content-Type": "application/json"
            },
            body: JSON.stringify({
              text: pageContent,
              query: query,
              session_id: tabs[0].id.toString()
            })
          });

          if (!response.ok) {
            throw new Error(`Server returned status ${response.status}`);
          }

          const data = await response.json();
          streamText(data.answer, responseBox);
        } catch (error) {
          responseBox.innerHTML = `<div class="error-message">⚠️ Connection Error: ${error.message}. Please verify that your Python server is running on port 8000.</div>`;
        } finally {
          // 2. Reset Button State
          askBtn.disabled = false;
          askBtn.innerText = "Ask";
        }
      }
    );
  });
});

// Submit on Enter key (Shift + Enter goes to a new line)
document.getElementById("userQuery").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault(); // Stop default newline behavior
    document.getElementById("askBtn").click(); // Trigger button click logic
  }
});

function getPageContent() {
  return document.body.innerText;
}

// Simulates a typewriter/fade-in effect word by word
function streamText(text, container) {
  container.innerHTML = '<div class="response-text"></div>';
  const textHolder = container.querySelector(".response-text");
  
  // Split the response by spaces, preserving newlines
  const words = text.split(/(\s+)/);
  let wordIndex = 0;

  function printNextWord() {
    if (wordIndex < words.length) {
      const word = words[wordIndex];
      
      if (word.trim() === "") {
        // It's a space or newline, append it directly
        textHolder.appendChild(document.createTextNode(word));
      } else {
        // Create a span with the animation class
        const span = document.createElement("span");
        span.className = "word";
        span.innerText = word;
        textHolder.appendChild(span);
      }
      
      wordIndex++;
      
      // Auto-scroll to the bottom as new words appear
      container.scrollTop = container.scrollHeight;
      
      // Control typing speed (40ms delay between words)
      setTimeout(printNextWord, 40);
    }
  }
  
  printNextWord();
}
