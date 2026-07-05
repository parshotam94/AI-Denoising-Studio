document.addEventListener("DOMContentLoaded", () => {
    setupNavigation();
    if (document.getElementById("btn-generate")) {
        document.getElementById("btn-generate").addEventListener("click", executePipeline);
    }
    if (document.getElementById("start-train-sim")) {
        document.getElementById("start-train-sim").addEventListener("click", runTrainingSimulation);
    }
    if (document.getElementById("latent-inspect-trigger")) {
        loadModelMetadata();
    }
});

function setupNavigation() {
    const currentPath = window.location.pathname;
    document.querySelectorAll(".nav-links a").forEach(link => {
        if (link.getAttribute("href") === currentPath || (currentPath === "/" && link.getAttribute("href") === "/")) {
            link.classList.add("active");
        }
    });
}

// Pipeline Steps for Dashboard Execution Automation
const STEPS = ["load", "noise", "encode", "latent", "decode", "metrics"];
function updatePipelineUI(activeStepId) {
    STEPS.forEach(step => {
        const el = document.getElementById(`step-${step}`);
        if (el) {
            el.classList.remove("active");
            if(step === activeStepId) el.classList.add("active");
        }
    });
}

async function executePipeline() {
    const digit = parseInt(document.getElementById("digit-select").value);
    const noiseLevel = parseFloat(document.getElementById("noise-slider").value);
    const generateBtn = document.getElementById("btn-generate");
    
    generateBtn.disabled = true;
    
    try {
        // Step 1: Loading
        updatePipelineUI("load");
        await new Promise(r => setTimeout(r, 400));
        
        // Step 2: Noise Injection
        updatePipelineUI("noise");
        await new Promise(r => setTimeout(r, 400));
        
        // Step 3-5: Send to backend for full processing
        updatePipelineUI("encode");
        const response = await fetch("/api/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ digit, noise_level: noiseLevel })
        });
        
        if (!response.ok) throw new Error("Backend inference cycle faulted.");
        const data = await response.json();
        
        updatePipelineUI("latent");
        await new Promise(r => setTimeout(r, 300));
        
        updatePipelineUI("decode");
        await new Promise(r => setTimeout(r, 300));
        
        updatePipelineUI("metrics");
        
        // Render to Screen Canvases
        renderMatrixToCanvas("canvas-orig", data.original);
        renderMatrixToCanvas("canvas-noisy", data.noisy);
        renderMatrixToCanvas("canvas-denoised", data.denoised);
        
        // Assign Metrics data to display elements
        document.getElementById("val-mse").innerText = data.metrics.mse;
        document.getElementById("val-psnr").innerText = data.metrics.psnr + " dB";
        document.getElementById("val-ssim").innerText = data.metrics.ssim;
        document.getElementById("val-time").innerText = data.metrics.inference_time_ms + " ms";
        document.getElementById("val-conf").innerText = `${data.classification.predicted_digit} (${data.classification.confidence}%)`;
        
    } catch (err) {
        alert("Pipeline error: " + err.message);
    } finally {
        generateBtn.disabled = false;
    }
}

function renderMatrixToCanvas(canvasId, matrix) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    const imgData = ctx.createImageData(28, 28);
    
    for (let i = 0; i < 28; i++) {
        for (let j = 0; j < 28; j++) {
            const val = Math.floor(matrix[i][j] * 255);
            const index = (i * 28 + j) * 4;
            imgData.data[index] = val;     // R
            imgData.data[index + 1] = val; // G
            imgData.data[index + 2] = val; // B
            imgData.data[index + 3] = 255; // A
        }
    }
    ctx.putImageData(imgData, 0, 0);
}

// Model Analysis Structure Metadata Loader
async function loadModelMetadata() {
    try {
        const response = await fetch("/api/model-info");
        const data = await response.json();
        
        document.getElementById("enc-layers").innerHTML = data.encoder_layers.map(l => `<li>${l}</li>`).join("");
        document.getElementById("latent-dim").innerText = JSON.stringify(data.latent_space_shape);
        document.getElementById("dec-layers").innerHTML = data.decoder_layers.map(l => `<li>${l}</li>`).join("");
    } catch (e) {
        console.error("Could not fetch architectural metadata");
    }
}

// Automated Simulation Tracker for Training Progression Dashboard
function runTrainingSimulation() {
    const consoleEl = document.getElementById("console-logs");
    const progressEl = document.getElementById("train-progress");
    if (!consoleEl) return;
    
    consoleEl.innerHTML = "> Establishing pipeline configurations...<br>";
    let epoch = 1;
    const maxEpochs = 5;
    
    const interval = setInterval(() => {
        if (epoch > maxEpochs) {
            clearInterval(interval);
            consoleEl.innerHTML += `><br>> Model state checkpoint verified. Exporting artifacts to /models/...<br>> Process Complete. Ready to deploy.`;
            progressEl.style.width = "100%";
            return;
        }
        
        const loss = (0.25 / epoch + Math.random() * 0.02).toFixed(4);
        const valLoss = (0.27 / epoch + Math.random() * 0.02).toFixed(4);
        
        consoleEl.innerHTML += `> Epoch ${epoch}/${maxEpochs} - Loss: ${loss} - Val_Loss: ${valLoss}<br>`;
        consoleEl.scrollTop = consoleEl.scrollHeight;
        
        progressEl.style.width = `${(epoch / maxEpochs) * 100}%`;
        epoch++;
    }, 1200);
}