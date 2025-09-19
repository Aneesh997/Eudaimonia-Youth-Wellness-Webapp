// Setup Firebase SDK
import { firebaseConfig } from "./firebaseConfig.js";
import { initializeApp } from "https://www.gstatic.com/firebasejs/11.1.0/firebase-app.js";
import {
  getAuth,
  onAuthStateChanged,
  signOut,
} from "https://www.gstatic.com/firebasejs/11.1.0/firebase-auth.js";
import {
  getDatabase,
  ref,
  get,
  set,
} from "https://www.gstatic.com/firebasejs/11.1.0/firebase-database.js";

// Initialize Firebase
const app = initializeApp(firebaseConfig);
const auth = getAuth(app);
const database = getDatabase(app);

// Fungsi untuk memperbarui tampilan profil pengguna
document.addEventListener("DOMContentLoaded", () => {
  onAuthStateChanged(auth, (user) => {
    if (user) {
      const userRef = ref(database, `users/${user.uid}`);
      get(userRef)
        .then((snapshot) => {
          if (snapshot.exists()) {
            updateUserProfile(snapshot.val());
          } else {
            console.warn("Data pengguna tidak ditemukan di database.");
          }
        })
        .catch((error) => {
          console.error("Error saat mengambil data pengguna:", error);
        });
    } else {
      console.log(
        "Pengguna tidak terautentikasi, mengalihkan ke halaman login."
      );
      window.location.href = "auth.html";
    }
  });
});

function updateUserProfile(userData) {
  if (!userData) {
    // console.error("Data pengguna tidak tersedia.");
    return;
  }

  const userNameElement = document.getElementById("userName");
  if (userNameElement) {
    userNameElement.textContent = userData.name || "Nama tidak tersedia";
  } else {
    // console.error("Element with ID 'userName' not found.");
  }

  const fullNameElement = document.getElementById("fullName");
  if (fullNameElement) {
    fullNameElement.textContent = userData.name || "Nama tidak tersedia";
  } else {
    // console.error("Element with ID 'fullName' not found.");
  }

  const userEmailElement = document.getElementById("userEmail");
  if (userEmailElement) {
    userEmailElement.textContent = userData.email || "Email tidak tersedia";
  } else {
    // console.error("Element with ID 'userEmail' not found.");
  }

  const userTtlElement = document.getElementById("userTtl");
  if (userTtlElement) {
    userTtlElement.textContent = userData.ttl || "Tanggal lahir tidak tersedia";
  } else {
    // console.error("Element with ID 'userTtl' not found.");
  }

  const userNohpElement = document.getElementById("userNohp");
  if (userNohpElement) {
    userNohpElement.textContent = userData.nohp || "Nomor HP tidak tersedia";
  } else {
    // console.error("Element with ID 'userNohp' not found.");
  }

  const lastLoginElement = document.getElementById("lastLogin");
  if (lastLoginElement) {
    lastLoginElement.textContent =
      userData.last_login || "Last Login tidak tersedia";
  } else {
    // console.error("Element with ID 'lastLogin' not found.");
  }
}

// Fungsi untuk logout pengguna dari aplikasi
document.getElementById("signout-button").addEventListener("click", () => {
  signOut(auth)
    .then(() => {
      // Alihkan ke halaman awal
      window.location.href = "index.html";
    })
    .catch((error) => {
      // console.error("Gagal keluar:", error);
    });
});

// Fungsi untuk menyimpan hasil quiz ke database
function saveQuizResult(score, percentage, userAnswers) {
  const user = auth.currentUser;

  if (user) {
    const uid = user.uid;
    const userRef = ref(database, `users/${uid}`);
    const quizResultRef = ref(database, `mentalResults/${uid}`);

    // Ambil data nama pengguna dari database
    get(userRef).then((snapshot) => {
      const name = snapshot.exists()
        ? snapshot.val().name
        : user.displayName || "Nama Tidak Tersedia";

      const mentalResults = {
        name: name,
        score: score,
        percentage: percentage,
        userAnswers: userAnswers,
        timestamp: new Date().toISOString(),
      };

      set(quizResultRef, mentalResults)
        .then(() => {
          // console.log("Hasil quiz berhasil disimpan ke database.");
          // console.log("Data yang dikirim ke Firebase:", mentalResults);
          showPopup("Hasil test berhasil disimpan.");
        })
        .catch((error) => {
          showPopup("Gagal menyimpan hasil test:");
        });
    });
  } else {
    console.error("Pengguna tidak login. Hasil quiz tidak dapat disimpan.");
  }
}

export { saveQuizResult };

// Fungsi untuk menyimpan hasil quiz ke database
function saveTestResult(score, percentage, userAnswers) {
  const user = auth.currentUser;

  if (user) {
    const uid = user.uid;
    const userRef = ref(database, `users/${uid}`);
    const testResultRef = ref(database, `stressResults/${uid}`);

    // Ambil data nama pengguna dari database
    get(userRef).then((snapshot) => {
      const name = snapshot.exists()
        ? snapshot.val().name
        : user.displayName || "Nama Tidak Tersedia";

      const stressResults = {
        name: name,
        score: score,
        percentage: percentage,
        userAnswers: userAnswers,
        timestamp: new Date().toISOString(),
      };

      set(testResultRef, stressResults)
        .then(() => {
          // console.log("Hasil quiz berhasil disimpan ke database.");
          // console.log("Data yang dikirim ke Firebase:", stressResults);
          showPopup("Hasil test berhasil disimpan.");
        })
        .catch((error) => {
          showPopup("Gagal menyimpan hasil test");
        });
    });
  } else {
    console.error("Pengguna tidak login. Hasil quiz tidak dapat disimpan.");
  }
}

export { saveTestResult };

// Helper: Show popup messages
function showPopup(message) {
  let popup = document.createElement("div");
  popup.innerText = message;
  popup.className = "popup";
  popup.style.display = "block";
  document.body.appendChild(popup);
  setTimeout(() => {
    document.body.removeChild(popup);
  }, 3000);
}
