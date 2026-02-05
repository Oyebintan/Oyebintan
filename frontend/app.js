const trainButton = document.getElementById("train-btn");
const predictButton = document.getElementById("predict-btn");
const trainOutput = document.getElementById("train-output");
const resultsContainer = document.getElementById("results");
const messageInput = document.getElementById("message-input");
const epochsInput = document.getElementById("epochs");

const apiRequest = async (url, payload) => {
  const response = await fetch(url, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });

  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.detail || "Request failed");
  }
  return data;
};

trainButton.addEventListener("click", async () => {
  trainOutput.textContent = "Training...";
  try {
    const epochs = Number(epochsInput.value || 12);
    const data = await apiRequest("/api/train", { epochs });
    trainOutput.textContent = JSON.stringify(data, null, 2);
  } catch (error) {
    trainOutput.textContent = error.message;
  }
});

predictButton.addEventListener("click", async () => {
  resultsContainer.innerHTML = "";
  try {
    const message = messageInput.value.trim();
    if (!message) {
      resultsContainer.textContent = "Please provide a message to classify.";
      return;
    }
    const data = await apiRequest("/api/predict", { messages: [message] });
    data.results.forEach((result) => {
      const card = document.createElement("div");
      card.className = "result-card";
      card.innerHTML = `
        <h3>${result.label.toUpperCase()}</h3>
        <p>Spam probability: ${(result.spam_probability * 100).toFixed(1)}%</p>
      `;
      resultsContainer.appendChild(card);
    });
  } catch (error) {
    resultsContainer.textContent = error.message;
  }
});
