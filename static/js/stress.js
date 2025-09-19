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
        let timer = 900; // 15 minutes in seconds
        let timerInterval;

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
        const restartButton = document.getElementById("restart-button");
        const infoButton = document.getElementById("info-button");

        // Initialize the quiz
        function initQuiz() {
            loadQuestion(currentQuestion);
            updateProgress();
            startTimer();
            
            // Add event listeners
            nextButton.addEventListener("click", handleNext);
            backButton.addEventListener("click", handleBack);
            restartButton.addEventListener("click", restartQuiz);
            infoButton.addEventListener("click", () => {
                window.location.href = 'materi-mental.html';
            });
        }

        function loadQuestion(index) {
            const question = questions[index];
            questionContainer.innerHTML = `
                <h5 class="mb-3">Test ${index + 1}</h5>
                <p>${question.question}</p>
                ${question.options.map((option, i) => `
                    <div class="form-check mb-2">
                        <input class="form-check-input" type="radio" name="question" id="q${index + 1}o${i}" value="${question.points[i]}" ${userAnswers[index] === question.points[i] ? 'checked' : ''}>
                        <label class="form-check-label" for="q${index + 1}o${i}">${option}</label>
                    </div>
                `).join("")}
            `;
            
            // Show/hide back button based on current question
            backButton.classList.toggle("d-none", index === 0);

            // Update next button text and style for last question
            if (index === questions.length - 1) {
                nextButton.textContent = "Finish";
                nextButton.classList.replace("btn-next", "btn-finish");
            } else {
                nextButton.textContent = "Next →";
                if (nextButton.classList.contains("btn-finish")) {
                    nextButton.classList.replace("btn-finish", "btn-next");
                }
            }
        }

        function updateProgress() {
            const progressPercentage = ((currentQuestion + 1) / questions.length) * 100;
            progressBar.style.width = `${progressPercentage}%`;
            questionProgress.textContent = `Test ${currentQuestion + 1} out of ${questions.length}`;
        }

        function startTimer() {
            clearInterval(timerInterval);
            updateTimerDisplay();
            
            timerInterval = setInterval(() => {
                if (timer > 0) {
                    timer--;
                    updateTimerDisplay();
                } else {
                    clearInterval(timerInterval);
                    finishQuiz();
                }
            }, 1000);
        }

        function updateTimerDisplay() {
            const minutes = Math.floor(timer / 60);
            const seconds = timer % 60;
            timerElement.textContent = `${minutes}:${seconds.toString().padStart(2, '0')}`;
            
            // Change color when time is running out
            if (timer < 60) {
                timerElement.classList.add('text-danger');
            } else {
                timerElement.classList.remove('text-danger');
            }
        }

        function handleNext() {
            const selectedOption = document.querySelector("input[name='question']:checked");
            
            if (!selectedOption) {
                alert("Please select an answer before proceeding.");
                return;
            }
            
            userAnswers[currentQuestion] = parseInt(selectedOption.value);
            
            if (currentQuestion < questions.length - 1) {
                currentQuestion++;
                loadQuestion(currentQuestion);
                updateProgress();
            } else {
                finishQuiz();
            }
        }

        function handleBack() {
            const selectedOption = document.querySelector("input[name='question']:checked");
            if (selectedOption) {
                userAnswers[currentQuestion] = parseInt(selectedOption.value);
            }
            
            if (currentQuestion > 0) {
                currentQuestion--;
                loadQuestion(currentQuestion);
                updateProgress();
            }
        }

        function calculateScore() {
            return userAnswers.reduce((total, points) => total + (points || 0), 0);
        }

        function finishQuiz() {
            clearInterval(timerInterval);
            
            const points = calculateScore();
            const percentage = Math.round((points / (questions.length * 3)) * 100);
            
            pointElement.textContent = `${points}`;
            scoreElement.textContent = `${percentage}%`;
            
            let diagnosis = "";
            let imageUrl = "";
            
            if (points <= 9) {
                diagnosis = "Tidak ada gejala depresi atau kecemasan.";
                imageUrl = "https://images.unsplash.com/photo-1506126613408-eca07ce68773?ixlib=rb-1.2.1&auto=format&fit=crop&w=500&q=60";
            } else if (points <= 19) {
                diagnosis = "Gejala ringan.";
                imageUrl = "https://images.unsplash.com/photo-1573497019940-1c28c88b4f3e?ixlib=rb-1.2.1&auto=format&fit=crop&w=500&q=60";
            } else if (points <= 29) {
                diagnosis = "Gejala sedang. Segera klik informasi penanganan mental untuk mengetahui lebih lanjut.";
                imageUrl = "https://images.unsplash.com/photo-1573497019940-1c28c88b4f3e?ixlib=rb-1.2.1&auto=format&fit=crop&w=500&q=60";
            } else if (points <= 45) {
                diagnosis = "Gejala berat. Segera klik informasi penanganan mental untuk mengetahui lebih lanjut.";
                imageUrl = "https://images.unsplash.com/photo-1545235617-9465d2a55698?ixlib=rb-1.2.1&auto=format&fit=crop&w=500&q=60";
            }

            diagnosisElement.textContent = diagnosis;
            resultImageElement.src = imageUrl;
            resultImageElement.alt = "Result Image";

            // Show result container, hide question container
            questionContainer.parentElement.parentElement.classList.add("d-none");
            resultContainer.classList.remove("d-none");
            
            // Save results (you can implement the Firebase saving function here)
            // saveQuizResult(points, percentage, userAnswers);
        }

        function restartQuiz() {
            currentQuestion = 0;
            userAnswers.fill(null);
            timer = 900;
            
            // Hide result container, show question container
            resultContainer.classList.add("d-none");
            questionContainer.parentElement.parentElement.classList.remove("d-none");
            
            // Reset progress bar
            progressBar.style.width = `${((currentQuestion + 1) / questions.length) * 100}%`;
            questionProgress.textContent = `Test ${currentQuestion + 1} out of ${questions.length}`;
            
            // Reload first question
            loadQuestion(currentQuestion);
            
            // Restart timer
            startTimer();
        }

// Initialize the quiz when the page loads
        document.addEventListener('DOMContentLoaded', initQuiz);

        // Initialize the quiz when the page loads