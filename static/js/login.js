// Setup Firebase SDK
import { firebaseConfig } from "./firebaseConfig.js";
import { initializeApp } from "https://www.gstatic.com/firebasejs/11.1.0/firebase-app.js";
import {
  getAuth,
  createUserWithEmailAndPassword,
  signInWithEmailAndPassword,
  GoogleAuthProvider,
  signInWithPopup,
  sendEmailVerification,
} from "https://www.gstatic.com/firebasejs/11.1.0/firebase-auth.js";
import {
  getDatabase,
  set,
  ref,
  update,
} from "https://www.gstatic.com/firebasejs/11.1.0/firebase-database.js";

// Initialize Firebase
const app = initializeApp(firebaseConfig);
const auth = getAuth(app);
const database = getDatabase(app);

// Get buttons
let signinButton = document.getElementById("signin-button");
let signupButton = document.getElementById("signup-button");

// Sign-up functionality
signupButton.addEventListener("click", (e) => {
  e.preventDefault();

  // Get input values
  let name = document.getElementById("name").value;
  let ttl = document.getElementById("ttl").value;
  let nohp = document.getElementById("nohp").value;
  let emailSignup = document.getElementById("email_signup").value;
  let passwordSignup = document.getElementById("psw_signup").value;

  // Input validation
  if (!name || !ttl || !nohp || !emailSignup || !passwordSignup) {
    showPopup("Semua kolom wajib diisi!");
    return;
  }

  if (!emailSignup.match(/^\S+@\S+\.\S+$/)) {
    showPopup("Format email tidak valid!");
    return;
  }

  if (passwordSignup.length < 6) {
    showPopup("Password harus minimal 6 karakter!");
    return;
  }

  // Create user with email and password
  createUserWithEmailAndPassword(auth, emailSignup, passwordSignup)
    .then((userCredential) => {
      const user = userCredential.user;

      // Save user data to Realtime Database
      set(ref(database, "users/" + user.uid), {
        name: name,
        ttl: ttl,
        nohp: nohp,
        email: emailSignup,
      })
        .then(() => {
          showPopup("Berhasil membuat akun, silahkan login!");
          setTimeout(() => {
            window.location.reload();
          }, 3000);
        })
        .catch((error) => {
          showPopup("Gagal menyimpan data: " + error.message);
        });
    })
    .catch((error) => {
      let errorMessage = "Gagal membuat akun";
      if (error.code === "auth/email-already-in-use") {
        errorMessage = "Email sudah terdaftar";
      } else if (error.code === "auth/invalid-email") {
        errorMessage = "Email tidak valid";
      } else if (error.code === "auth/weak-password") {
        errorMessage = "Password minimal 6 karakter";
      }
      showPopup(errorMessage);
    });
});

// Sign-in functionality
signinButton.addEventListener("click", (e) => {
  e.preventDefault();

  let emailSignin = document.getElementById("email_signin").value;
  let passwordSignin = document.getElementById("psw_signin").value;

  if (!emailSignin || !passwordSignin) {
    showPopup("Email dan password wajib diisi!");
    return;
  }

  signInWithEmailAndPassword(auth, emailSignin, passwordSignin)
    .then((userCredential) => {
      const user = userCredential.user;

      let lgDate = new Date();
      update(ref(database, "users/" + user.uid), {
        last_login: lgDate,
      })
        .then(() => {
          showPopup("Login berhasil!");
          setTimeout(() => {
            window.location.href = "home.html";
          }, 1200);
        })
        .catch((error) => {
          showPopup("Gagal memperbarui data login: " + error.message);
        });
    })
    .catch((error) => {
      let errorMessage = "Login gagal!";
      if (error.code === "auth/user-not-found") {
        errorMessage = "Pengguna tidak ditemukan!";
      } else if (error.code === "auth/invalid-credential") {
        errorMessage = "Password salah!";
      } else if (error.code === "auth/invalid-email") {
        errorMessage = "Email tidak valid!";
      } else if (error.code === "auth/too-many-requests") {
        errorMessage = "Terlalu banyak percobaan login! Coba lagi nanti.";
      } else {
        errorMessage = error.message;
      }
      showPopup(errorMessage);
    });
});

// Google Sign-In Functionality
document
  .getElementById("googleSignInBtn")
  .addEventListener("click", async () => {
    try {
      const provider = new GoogleAuthProvider();
      const result = await signInWithPopup(auth, provider);
      const user = result.user;

      const userData = {
        uid: user.uid,
        name: user.displayName,
        email: user.email,
        photoURL: user.photoURL,
        emailVerified: user.emailVerified,
        lastLogin: new Date().toISOString(),
      };

      await set(ref(database, "users/" + userData.uid), userData);

      if (!userData.emailVerified) {
        await sendEmailVerification(user);
        showPopup(`Email verifikasi telah dikirim ke ${userData.email}.`);
      } else {
        showPopup(`Login Berhasil!`, 1000);
        setTimeout(() => (window.location.href = "home.html"), 1000);
      }
    } catch (error) {
      showPopup("Login dengan Google gagal!");
    }
  });

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
