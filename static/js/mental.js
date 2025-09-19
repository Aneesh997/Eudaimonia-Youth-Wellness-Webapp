import { saveQuizResult } from "./getdb.js";

const questions = [
    {
        question: "Pernahkah Anda mengalami rasa kurang minat dari aktivitas yang biasanya Anda lakukan?",
        options: ["Tidak Pernah", "Beberapa Hari", "Sebagian Besar Hari", "Setiap Hari"],
        points: [0, 1, 2, 3]
    },
    {
        question: "Apakah anda merasa ada perasaan tidak nyaman ketika berada di sekitar orang lain?",
        options: ["Tidak Pernah", "Beberapa Hari", "Sebagian Besar Hari", "Setiap Hari"],
        points: [0, 1, 2, 3]
    },
    {
        question: "Apakah kebiasaan pola makan anda berubah drastis?",
        options: ["Tidak Pernah", "Beberapa Hari", "Sebagian Besar Hari", "Setiap Hari"],
        points: [0, 1, 2, 3]
    },
    {
        question: "Seberapa sering anda merasa terganggu karena tidak bisa berhenti khawatir?",
        options: ["Tidak Pernah", "Beberapa Hari", "Sebagian Besar Hari", "Setiap Hari"],
        points: [0, 1, 2, 3]
    },
    {
        question: "Seberapa sering anda merasa tidak puas dengan diri anda sendiri selama 1 bulan terakhir?",
        options: ["Tidak Pernah", "Beberapa Hari", "Sebagian Besar Hari", "Setiap Hari"],
        points: [0, 1, 2, 3]
    },
    {
        question: "Seberapa sering selama beberapa minggu terakhir anda merasa masa depan suram?",
        options: ["Tidak Pernah", "Beberapa Hari", "Sebagian Besar Hari", "Setiap Hari"],
        points: [0, 1, 2, 3]
    },
    {
        question: "Seberapa sering anda merasa terganggu perasaan sedih atau tertekan?",
        options: ["Tidak Pernah", "Beberapa Hari", "Sebagian Besar Hari", "Setiap Hari"],
        points: [0, 1, 2, 3]
    },
    {
        question: "Selama 1 bulan terakhir, seberapa sering anda merasa sendirian atau kesepian?",
        options: ["Tidak Pernah", "Beberapa Hari", "Sebagian Besar Hari", "Setiap Hari"],
        points: [0, 1, 2, 3]
    },
    {
        question: "Selama 1 bulan terakhir, seberapa sering anda secara serius mempertimbangkan untuk mencoba bunuh diri?",
        options: ["Tidak Pernah", "Beberapa Hari", "Sebagian Besar Hari", "Setiap Hari"],
        points: [0, 1, 2, 3]
    },
    {
        question: "Pernahkah Anda merasa terpengaruh oleh perasaan gelisah, cemas, atau gugup?",
        options: ["Tidak Pernah", "Beberapa Hari", "Sebagian Besar Hari", "Setiap Hari"],
        points: [0, 1, 2, 3]
    },
    {
        question: "Seberapa sering Anda merasa kurang senang atau tertarik pada aktivitas yang biasanya Anda lakukan?",
        options: ["Tidak Pernah", "Beberapa Hari", "Sebagian Besar Hari", "Setiap Hari"],
        points: [0, 1, 2, 3]
    },
    {
        question: "Seberapa sering Anda melakukan hal-hal yang berarti bagi Anda atau kehidupan Anda?",
        options: ["Tidak Pernah", "Beberapa Hari", "Sebagian Besar Hari", "Setiap Hari"],
        points: [0, 1, 2, 3]
    },
    {
        question: "Seringkah anda merasa lebih baik sendiri daripada bergaul dengan teman-teman?",
        options: ["Tidak Pernah", "Beberapa Hari", "Sebagian Besar Hari", "Setiap Hari"],
        points: [0, 1, 2, 3]
    },
    {
        question: "Seringkah anda menangis dalam 1 minggu terakhir?",
        options: ["Tidak Pernah", "Beberapa Hari", "Sebagian Besar Hari", "Setiap Hari"],
        points: [0, 1, 2, 3]
    },
    {
        question: "Seringkah anda merasa menyalahkan diri sendiri ketika ada masalah?",
        options: ["Tidak Pernah", "Beberapa Hari", "Sebagian Besar Hari", "Setiap Hari"],
        points: [0, 1, 2, 3]
    }
];

let currentQuestion = 0;
const userAnswers = Array(questions.length).fill(null);
let timer = 900;

const questionContainer = document.getElementById("question-container");
const nextButton = document.getElementById("next-button");
const backButton = document.getElementById("back-button");
const progressBar = document.getElementById("progress-bar");
const questionProgress = document.getElementById("question-progress");
const resultContainer = document.getElementById("result-container");
const pointElement = document.getElementById("point");
const scoreElement = document.getElementById("score");
const timerElement = document.getElementById("timer");
const resultImageElement = document.getElementById("result-image");
const diagnosisElement = document.getElementById("diagnosis");


function loadQuestion(index) {
    const question = questions[index];
    questionContainer.innerHTML = `
        <h5 class="mb-3">Test ${index + 1}</h5>
        <p>${question.question}</p>
        ${question.options.map((option, i) => `
            <div class="form-check">
                <input class="form-check-input" type="radio" name="question" id="q${index + 1}o${i}" value="${question.points[i]}" ${userAnswers[index] === question.points[i] ? 'checked' : ''}>
                <label class="form-check-label" for="q${index + 1}o${i}">${option}</label>
            </div>
        `).join("")}
    `;
    backButton.classList.toggle("d-none", index === 0);

    if (index === questions.length - 1) {
        nextButton.textContent = "Finish";
        nextButton.classList.replace("btn-next", "btn-finish");
    } else {
        nextButton.textContent = "Next →";
        nextButton.classList.replace("btn-finish", "btn-next");
    }
}

function updateProgress() {
    progressBar.style.width = `${((currentQuestion + 1) / questions.length) * 100}%`;
    questionProgress.textContent = `Test ${currentQuestion + 1} out of ${questions.length}`;
}

function updateTimer() {
    const minutes = Math.floor(timer / 60);
    const seconds = timer % 60;
    timerElement.textContent = `${minutes}:${seconds.toString().padStart(2, '0')}`;
    if (timer > 0) {
        timer--;
    } else {
        finishQuiz();
    }
}

function calculateScore() {
    return userAnswers.reduce((total, points) => total + (points || 0), 0);
}

function finishQuiz() {
    const points = calculateScore();
    const percentage = Math.round((points / (questions.length * 3)) * 100);
    
    pointElement.textContent = `${points}`;
    scoreElement.textContent = `${percentage}%`;
    
    let diagnosis = "";
    let imageUrl = "";
    
    if (points <= 9) {
        diagnosis = "Tidak ada gejala depresi atau kecemasan.";
        imageUrl = "assets/img/low.png";
    } else if (points <= 19) {
        diagnosis = "Gejala ringan.";
        imageUrl = "assets/img/low.png";
    } else if (points <= 29) {
        diagnosis = "Gejala sedang. Segerea klik informasi penanganan mental untuk mengetahui lebih lanjut.";
        imageUrl = "assets/img/medium.png";
    } else if (points <= 45) {
        diagnosis = "Gejala berat. Segerea klik informasi penanganan mental untuk mengetahui lebih lanjut.";
        imageUrl = "assets/img/high.png";
    }

    diagnosisElement.textContent = diagnosis;
    resultImageElement.src = imageUrl;

    questionContainer.parentElement.classList.add("d-none");
    resultContainer.classList.remove("d-none");

    // Panggil fungsi untuk menyimpan hasil ke Firebase
    saveQuizResult(points, percentage, userAnswers);
}

nextButton.addEventListener("click", () => {
    const selectedOption = document.querySelector("input[name='question']:checked");
    if (selectedOption) {
        userAnswers[currentQuestion] = parseInt(selectedOption.value);
    }
    if (currentQuestion < questions.length - 1) {
        currentQuestion++;
        loadQuestion(currentQuestion);
        updateProgress();
    } else {
        finishQuiz();
    }
});

backButton.addEventListener("click", () => {
    const selectedOption = document.querySelector("input[name='question']:checked");
    if (selectedOption) {
        userAnswers[currentQuestion] = parseInt(selectedOption.value);
    }
    currentQuestion--;
    loadQuestion(currentQuestion);
    updateProgress();
});

window.restartQuiz = function() {
    currentQuestion = 0;
    userAnswers.fill(null);
    resultContainer.classList.add("d-none");
    questionContainer.parentElement.classList.remove("d-none");
    progressBar.style.width = `20%`;
    questionProgress.textContent = `Test 1 out of ${questions.length}`;
    loadQuestion(currentQuestion);
    timer = 900;
};


loadQuestion(currentQuestion);
updateProgress();
setInterval(updateTimer, 1000);