/*
document.addEventListener("DOMContentLoaded", () => {
    const input = document.getElementById("input");
    const output = document.getElementById("output");

    // Dark / Light Mode toggle
    document.getElementById("mode-toggle").onclick = () => {
        document.body.classList.toggle("dark-mode");
        document.body.classList.toggle("light-mode");
    };

    // Function to copy input content cleanly
    function copyContentForOutput() {
        if (!output || !input) return;

        // Create temporary container to strip inline styles
        const temp = document.createElement("div");
        temp.innerHTML = input.innerHTML;

        // Remove all inline font-family styles recursively
        temp.querySelectorAll("*").forEach(el => el.removeAttribute("style"));

        return temp.innerHTML;
    }

    // Lexend font button
    document.getElementById("lexend-btn").onclick = () => {
        output.innerHTML = copyContentForOutput(); // copy input content
        output.style.fontFamily = "'Lexend', sans-serif"; // apply font only to output
    };

    // OpenDyslexic font button
    document.getElementById("opendys-btn").onclick = () => {
        output.innerHTML = copyContentForOutput(); // copy input content
        output.style.fontFamily = "'OpenDyslexic', 'OpenDyslexicRegular', sans-serif";
    };

    // Letter spacing control
    document.getElementById("letter-space").oninput = (e) => {
        if (!output) return;
        output.style.letterSpacing = e.target.value + "px";
    };

    // Line spacing control
    document.getElementById("line-space").oninput = (e) => {
        if (!output) return;
        output.style.lineHeight = e.target.value;
    };



});
*/

document.addEventListener("DOMContentLoaded", () => {
    const input = document.getElementById("input");
    const output = document.getElementById("output");

    // === Dark / Light Mode ===
    document.getElementById("mode-toggle").onclick = () => {
        document.body.classList.toggle("dark-mode");
        document.body.classList.toggle("light-mode");
    };

    // === Helper: copy input content cleanly to output ===
    function copyContentForOutput() {
        if (!output || !input) return;

        // Create temporary container to strip inline styles
        const temp = document.createElement("div");
        temp.innerHTML = input.innerHTML;

        // Remove all inline styles to avoid overriding container font
        temp.querySelectorAll("*").forEach(el => el.removeAttribute("style"));

        return temp.innerHTML;
    }

    // === Font buttons ===
    document.getElementById("lexend-btn").onclick = () => {
        output.innerHTML = copyContentForOutput();
        output.style.fontFamily = "'Lexend', sans-serif";
    };

    document.getElementById("opendys-btn").onclick = () => {
        output.innerHTML = copyContentForOutput();
        output.style.fontFamily = "'OpenDyslexic', 'OpenDyslexicRegular', sans-serif";
    };

    // === Letter spacing & line height ===
    document.getElementById("letter-space").oninput = (e) => {
        if (!output) return;
        output.style.letterSpacing = e.target.value + "px";
    };

    document.getElementById("line-space").oninput = (e) => {
        if (!output) return;
        output.style.lineHeight = e.target.value;
    };

        // Speak highlighted text or full output
    document.getElementById("speak-btn").onclick = () => {
        const selection = window.getSelection();
        let textToSpeak = "";

        if (selection.rangeCount > 0) {
            const range = selection.getRangeAt(0);
            if (output.contains(range.commonAncestorContainer)) {
                textToSpeak = selection.toString();
            }
        }

        if (!textToSpeak) {
            textToSpeak = output.innerText;
        }

        if (textToSpeak) {
            const utterance = new SpeechSynthesisUtterance(textToSpeak);
            utterance.rate = 1;
            utterance.pitch = 1;
            speechSynthesis.speak(utterance);
        }
    };

    
});
