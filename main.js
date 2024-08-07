import {
    HandLandmarker,
    FilesetResolver
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0";

class SimpleKalmanFilter {
    constructor(R, Q) {
        this.R = R; // noise covariance
        this.Q = Q; // process covariance
        this.A = 1;
        this.B = 0;
        this.C = 1;
        this.cov = NaN;
        this.x = NaN; // estimated signal without noise
    }

    filter(z) {
        if (isNaN(this.x)) {
            this.x = (1 / this.C) * z;
            this.cov = (1 / this.C) * this.Q * (1 / this.C);
        } else {
            // Compute prediction
            let predX = (this.A * this.x) + (this.B * 0);
            let predCov = ((this.A * this.cov) * this.A) + this.R;

            // Kalman gain
            let K = predCov * this.C * (1 / ((this.C * predCov * this.C) + this.Q));

            // Correction
            this.x = predX + K * (z - (this.C * predX));
            this.cov = predCov - (K * this.C * predCov);
        }

        return this.x;
    }
}

const demosSection = document.getElementById("demos");

let handLandmarker = undefined;
let runningMode = "IMAGE";
let enableWebcamButton;
let captureButton;
let webcamRunning = false;

// Before we can use HandLandmarker class we must wait for it to finish loading.
const createHandLandmarker = async () => {
    const vision = await FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm"
    );
    handLandmarker = await HandLandmarker.createFromOptions(vision, {
        baseOptions: {
            modelAssetPath: `https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task`,
            delegate: "GPU"
        },
        runningMode: runningMode,
        numHands: 2
    });
    demosSection.classList.remove("invisible");
};
createHandLandmarker();

// Demo 1: Grab a bunch of images from the page and detect them upon click.
const imageContainers = document.getElementsByClassName("detectOnClick");

for (let i = 0; i < imageContainers.length; i++) {
    imageContainers[i].children[0].addEventListener("click", handleClick);
}

async function handleClick(event) {
    if (!handLandmarker) {
        console.log("Wait for handLandmarker to load before clicking!");
        return;
    }

    if (runningMode === "VIDEO") {
        runningMode = "IMAGE";
        await handLandmarker.setOptions({ runningMode: "IMAGE" });
    }

    const allCanvas = event.target.parentNode.getElementsByClassName("canvas");
    for (var i = allCanvas.length - 1; i >= 0; i--) {
        const n = allCanvas[i];
        n.parentNode.removeChild(n);
    }

    const handLandmarkerResult = await handLandmarker.detect(event.target);
    const canvas = document.createElement("canvas");
    canvas.setAttribute("class", "canvas");
    canvas.setAttribute("width", event.target.naturalWidth + "px");
    canvas.setAttribute("height", event.target.naturalHeight + "px");
    canvas.style =
        "left: 0px;" +
        "top: 0px;" +
        "width: " +
        event.target.width +
        "px;" +
        "height: " +
        event.target.height +
        "px;";

    event.target.parentNode.appendChild(canvas);
    const cxt = canvas.getContext("2d");

    for (const landmarks of handLandmarkerResult.landmarks) {
        drawConnectors(cxt, landmarks, HAND_CONNECTIONS, {
            color: "#00FF00",
            lineWidth: 5
        });
        drawLandmarks(cxt, landmarks, { color: "#FF0000", lineWidth: 2 });
    }
}

// Demo 2: Continuously grab image from webcam stream and detect it.
const videoHeight = "640px";
const videoWidth = "480px";

const liveView = document.getElementById("liveView");
const webcamElement = document.getElementById("webcam");
const canvasElement = document.getElementById("output_canvas");
const viewContainer = document.getElementById("viewContainer");
const canvasCtx = canvasElement.getContext("2d");

function hasGetUserMedia() {
    return !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
}

if (hasGetUserMedia()) {
    enableWebcamButton = document.getElementById("webcamButton");
    enableWebcamButton.addEventListener("click", enableCam);
    
    captureButton = document.getElementById("captureButton");
    captureButton.disabled = true; // Initially disable the Capture button
    captureButton.addEventListener("click", captureImage);
} else {
    console.warn("getUserMedia() is not supported by your browser");
}

async function enableCam(event) {
    if (!handLandmarker) {
        console.log("Wait! objectDetector not loaded yet.");
        return;
    }

    if (webcamRunning === true) {
        webcamRunning = false;
        enableWebcamButton.innerText = "ENABLE WEBCAM";
        webcamElement.srcObject.getVideoTracks().forEach((track) => {
            track.stop();
        });
        return;
    }

    runningMode = "VIDEO";
    await handLandmarker.setOptions({ runningMode: "VIDEO" });

    webcamRunning = true;
    enableWebcamButton.style.display = "none";

    const constraints = {
        video: {
            width: { ideal: 1920 },
            height: { ideal: 1080 },
            facingMode: { ideal: "environment" } // Prefer back camera
        }
    };

    navigator.mediaDevices.getUserMedia(constraints).then(function (stream) {
        webcamElement.srcObject = stream;

        const videoTracks = stream.getVideoTracks();
        if (videoTracks.length > 0 && videoTracks[0].getSettings().facingMode === 'environment') {
            webcamElement.classList.remove('flip');
            canvasElement.classList.remove('flip');
            canvasElement.classList.add('noflip');
        } else {
            webcamElement.classList.add('flip');
            canvasElement.classList.add('flip');
            canvasElement.classList.remove('noflip');
        }

        webcamElement.addEventListener("loadeddata", predictWebcam);
    });
}

const referencePositions = [
    { "x": 0.48430564999580383, "y": 0.8110854625701904, "z": 5.423733568932221e-7 },
    { "x": 0.5502023696899414, "y": 0.7348806858062744, "z": -0.00948019977658987 },
    { "x": 0.589587926864624, "y": 0.6451688408851624, "z": -0.021786006167531013 },
    { "x": 0.6157299876213074, "y": 0.5460792183876038, "z": -0.032154250890016556 },
    { "x": 0.6464489102363586, "y": 0.4790748953819275, "z": -0.04322601854801178 },
    { "x": 0.5311582684516907, "y": 0.5096843242645264, "z": -0.03053961880505085 },
    { "x": 0.5542333126068115, "y": 0.3711826205253601, "z": -0.04405640438199043 },
    { "x": 0.5643584728240967, "y": 0.2880013585090637, "z": -0.05069946125149727 },
    { "x": 0.5714240670204163, "y": 0.228133887052536, "z": -0.055220432579517365 },
    { "x": 0.48337316513061523, "y": 0.49954313039779663, "z": -0.03383808210492134 },
    { "x": 0.4870733618736267, "y": 0.3366149663925171, "z": -0.047285668551921844 },
    { "x": 0.4889812469482422, "y": 0.24255409836769104, "z": -0.05254475399851799 },
    { "x": 0.4909090995788574, "y": 0.17631328105926514, "z": -0.05623533949255943 },
    { "x": 0.44047462940216064, "y": 0.5183423161506653, "z": -0.03667522221803665 },
    { "x": 0.43307211995124817, "y": 0.36207687854766846, "z": -0.050330400466918945 },
    { "x": 0.43187040090560913, "y": 0.27007052302360535, "z": -0.059464603662490845 },
    { "x": 0.43410253524780273, "y": 0.20186573266983032, "z": -0.06592313200235367 },
    { "x": 0.40458112955093384, "y": 0.5548273324966431, "z": -0.038843441754579544 },
    { "x": 0.38064879179000854, "y": 0.4458256959915161, "z": -0.05402098968625069 },
    { "x": 0.3689345717430115, "y": 0.37663131952285767, "z": -0.06190962344408035 },
    { "x": 0.3627241551876068, "y": 0.31447169184684753, "z": -0.06592313200235367 }
];

function isHandFlat(landmarks) {
    
    const threshold = 0.05; // Define an appropriate threshold value

    // Indexes of landmarks to exclude (finger tips)
    const excludedIndexes = [4, 8, 12, 16, 20]; // Example indexes, adjust based on your landmark model

    for (let i = 0; i < landmarks.length; i++) {
        if (excludedIndexes.includes(i)) {
            continue; // Skip comparison for excluded landmarks (tips)
        }

        const landmark = landmarks[i];
        const reference = referencePositions[i];

        const dx = Math.abs(landmark.x - reference.x);
        const dy = Math.abs(landmark.y - reference.y);
        const dz = Math.abs(landmark.z - reference.z);

        if (dx > threshold || dy > threshold || dz > threshold) {
            return false; // Landmark is out of the threshold range
        }
    }

    return true; // All non-tip landmarks are within the threshold range
}

function isHandFlat1(landmarks, referenceLandmarks) {
    // Extract relevant landmarks for flatness check
    const landmarksToCheck = [
        landmarks[0], // Example: Wrist or palm center
        landmarks[1], // Example: Base of index finger
        landmarks[2], // Example: Base of middle finger
        landmarks[3], // Example: Base of ring finger
        landmarks[4]  // Example: Base of pinky
    ];

    // Define a threshold for acceptable z-coordinate variance
    const zThreshold = 0.02;

    // Extract z-coordinates from reference landmarks
    const referenceZs = referencePositions.map(l => l.z);

    // Check if the z-coordinates of the hand landmarks are within the threshold of the reference
    return landmarksToCheck.every((landmark, index) => {
        const referenceZ = referenceZs[index];
        return Math.abs(landmark.z - referenceZ) < zThreshold;
    });
}


async function predictWebcam() {
    canvasElement.width = webcamElement.videoWidth;
    canvasElement.height = webcamElement.videoHeight;

    const minSide = Math.min(canvasElement.width, canvasElement.height);
    const squareSize = minSide * 0.8;
    const squareX = (canvasElement.width - squareSize) / 2;
    const squareY = (canvasElement.height - squareSize) / 2;

    // Now let's start detecting the stream.
    if (webcamRunning === true) {
        const startTimeMs = performance.now();

        const handLandmarkerResult = await handLandmarker.detectForVideo(
            webcamElement,
            startTimeMs
        );

        canvasCtx.save();
        canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
        canvasCtx.drawImage(
            webcamElement,
            0,
            0,
            canvasElement.width,
            canvasElement.height
        );

        // Draw the square
        canvasCtx.strokeStyle = "#FFFF00";
        canvasCtx.lineWidth = 4;
        canvasCtx.strokeRect(squareX, squareY, squareSize, squareSize);

        let handDetected = false;
        let allLandmarksInSquare = false;
        let handFlat = false;

        if (handLandmarkerResult.landmarks) {
            for (const landmarks of handLandmarkerResult.landmarks) {
                drawConnectors(canvasCtx, landmarks, HAND_CONNECTIONS, {
                    color: "#00FF00",
                    lineWidth: 5
                });
                drawLandmarks(canvasCtx, landmarks, {
                    color: "#FF0000",
                    lineWidth: 2
                });

                handDetected = true;

                // Check if all landmarks are inside the square
                allLandmarksInSquare = landmarks.every(landmark => {
                    const x = landmark.x * canvasElement.width;
                    const y = landmark.y * canvasElement.height;
                    return (
                        x >= squareX &&
                        x <= squareX + squareSize &&
                        y >= squareY &&
                        y <= squareY + squareSize
                    );
                });

                // Check if the hand is flat
                handFlat = isHandFlat1(landmarks);

                if (allLandmarksInSquare) {
                    console.log("Hand is in the square!");
                    landmarksToSave = landmarks;
                }

                if (handFlat) {
                    console.log("Hand is in the flat!");
                }
            }

            
        }

        canvasCtx.restore();

        // Enable capture button only when hand is detected, all landmarks are in the square, and the hand is flat
        captureButton.disabled = !(handDetected && allLandmarksInSquare && handFlat);

        // Call this function again to keep predicting when the browser is ready
        window.requestAnimationFrame(predictWebcam);
    }
}

var landmarksToSave;

function captureImage() {

    console.log(landmarksToSave);

}
