// ---------------- DOM Elements ----------------
const chatbox = document.getElementById("chatbox");
const userInput = document.getElementById("userInput");
const sendBtn = document.getElementById("sendBtn");
const voiceBtn = document.getElementById("voiceBtn");
let sessionId = "default";

const settingsBtn = document.getElementById("settingsBtn");
const settingsModal = document.getElementById("settingsModal");
const closeSettings = document.getElementById("closeSettings");
const darkModeToggle = document.getElementById("darkModeToggle");
const saveSettingsBtn = document.getElementById("saveSettings");

// ---------------- Append message ----------------
function appendMessage(content, sender) {
    if (!content || content.trim() === "") return;
    const msgDiv = document.createElement("div");
    msgDiv.classList.add("message", sender);
    msgDiv.innerText = content;
    chatbox.appendChild(msgDiv);
    chatbox.scrollTop = chatbox.scrollHeight;
}

// ---------------- Typing indicator ----------------
function showTyping() {
    const typingDiv = document.createElement("div");
    typingDiv.classList.add("message", "bot", "typing");
    typingDiv.id = "typingIndicator";
    typingDiv.innerText = "🤖 Bot is typing...";
    chatbox.appendChild(typingDiv);
    chatbox.scrollTop = chatbox.scrollHeight;
}

function removeTyping() {
    const typingDiv = document.getElementById("typingIndicator");
    if (typingDiv) typingDiv.remove();
}

// ---------------- Speak bot response ----------------
function speakText(text) {
    if ("speechSynthesis" in window && text.trim() !== "") {
        speechSynthesis.cancel();
        const utterance = new SpeechSynthesisUtterance(text);
        utterance.lang = "en-US";
        speechSynthesis.speak(utterance);
    }
}

// ---------------- Get response from backend ----------------
async function getBotResponse(message) {
    showTyping();
    try {
        const res = await fetch("/api/chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ message, session_id: sessionId })
        });
        removeTyping();
        if (!res.ok) throw new Error("Network error: " + res.status);
        const data = await res.json();
        return data.reply || "⚠️ No response from server.";
    } catch (err) {
        removeTyping();
        console.error("Fetch error:", err);
        return "❌ Error: Unable to connect to server.";
    }
}

// ---------------- Send message ----------------
async function sendMessage(text = null) {
    const message = text || userInput.value.trim();
    if (!message) return;

    appendMessage(message, "user");
    userInput.value = "";
    userInput.focus();

    // Appointment shortcut
    if (message.toLowerCase().includes("appointment")) {
        appendMessage("📅 Redirecting to appointment page...", "bot");
        speakText("Redirecting to appointment page");
        setTimeout(() => window.location.href = "/appointment", 1000);
        return;
    }

    const botResponse = await getBotResponse(message);
    appendMessage(botResponse, "bot");
    speakText(botResponse);
}

// ---------------- Events ----------------
sendBtn.addEventListener("click", () => sendMessage());
userInput.addEventListener("keypress", e => { if (e.key === "Enter") sendMessage(); });

// ---------------- Voice recognition ----------------
voiceBtn.addEventListener("click", () => {
    if (!("webkitSpeechRecognition" in window) && !("SpeechRecognition" in window)) {
        appendMessage("⚠️ Speech recognition not supported.", "bot");
        return;
    }

    const Recognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    const recognition = new Recognition();
    recognition.lang = "en-US";
    recognition.interimResults = false;  // Only final results
    recognition.maxAlternatives = 1;

    const placeholderMsg = document.createElement("div");
    placeholderMsg.classList.add("message", "bot");
    placeholderMsg.innerText = "🎙️ Listening...";
    chatbox.appendChild(placeholderMsg);
    chatbox.scrollTop = chatbox.scrollHeight;

    recognition.start();

    // Collect all final transcripts
    let finalTranscript = "";

    recognition.onresult = event => {
        for (let i = event.resultIndex; i < event.results.length; i++) {
            if (event.results[i].isFinal) {
                finalTranscript += event.results[i][0].transcript + " ";
            }
        }
    };

    recognition.onerror = event => {
        if (placeholderMsg.parentElement) placeholderMsg.remove();
        appendMessage("❌ Voice recognition error: " + event.error, "bot");
        speakText("Voice recognition error");
    };

    recognition.onend = () => {
        if (placeholderMsg.parentElement) placeholderMsg.remove();
        if (finalTranscript.trim() !== "") {
            sendMessage(finalTranscript.trim());
        }
    };
});

// ---------------- Settings Modal ----------------
settingsBtn.addEventListener("click", () => settingsModal.style.display = "block");
closeSettings.addEventListener("click", () => settingsModal.style.display = "none");
window.addEventListener("click", e => { if (e.target === settingsModal) settingsModal.style.display = "none"; });

// ---------------- Dark Mode toggle ----------------
darkModeToggle.addEventListener("change", () => {
    const isDark = darkModeToggle.checked;
    document.body.classList.add("switching-dark");
    document.body.classList.toggle("dark-mode", isDark);
    localStorage.setItem("darkMode", isDark);

    const msg = isDark ? "🌙 Switching to Dark Mode..." : "☀️ Switching to Light Mode...";
    appendMessage(msg, "bot");
    speakText(msg);

    setTimeout(() => document.body.classList.remove("switching-dark"), 700);
});

// ---------------- Save profile info ----------------
saveSettingsBtn.addEventListener("click", () => {
    const profile = {
        name: document.getElementById("userName").value.trim(),
        age: document.getElementById("userAge").value.trim(),
        gender: document.getElementById("userGender").value,
        health: document.getElementById("userHealth").value.trim()
    };
    localStorage.setItem("profile", JSON.stringify(profile));

    const msg = `✅ Profile saved! Hello, ${profile.name || "User"}!`;
    appendMessage(msg, "bot");
    speakText(msg);

    settingsModal.style.display = "none";

    // Update greeting if chatbox is empty
    if (chatbox.childElementCount === 0) {
        const greetMsg = `👋 Hello, ${profile.name || "User"}! How can I assist you today?`;
        appendMessage(greetMsg, "bot");
        speakText(greetMsg);
    }
});

// ---------------- Load saved settings ----------------
window.addEventListener("load", () => {
    // Load dark mode
    const darkMode = localStorage.getItem("darkMode") === "true";
    darkModeToggle.checked = darkMode;
    if (darkMode) document.body.classList.add("dark-mode");

    // Load profile info
    const profile = JSON.parse(localStorage.getItem("profile") || "{}");
    if (profile.name) document.getElementById("userName").value = profile.name;
    if (profile.age) document.getElementById("userAge").value = profile.age;
    if (profile.gender) document.getElementById("userGender").value = profile.gender;
    if (profile.health) document.getElementById("userHealth").value = profile.health;

    // Greet user
    const greetMsg = profile.name
        ? `👋 Hello, ${profile.name}! How can I assist you today?`
        : "👋 Hello, User! How can I assist you today?";
    appendMessage(greetMsg, "bot");
    speakText(greetMsg);
});
