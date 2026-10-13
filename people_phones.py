"""
celular_personas.py  (TODO EN UN SOLO ARCHIVO)
==============================================
La cámara del CELULAR se ve en el COMPUTADOR y el programa solo dice
(en pantalla y en voz alta) cuántas personas hay:

    "No hay nadie"  /  "Hay una persona"  /  "Hay 2 personas"  ...

Cómo usarlo:
  1. Corre este archivo:   python celular_personas.py
  2. Te muestra un LINK (y un QR si tienes qrcode). Ábrelo en CHROME del celular,
     toca "Iniciar cámara" y acepta el permiso.
  3. En el celular NO se ve la cámara; la imagen sale en el computador.

Instalar una sola vez:
    pip install aiohttp opencv-python numpy cryptography
    pip install cvzone mediapipe pyttsx3
    pip install qrcode        (opcional: QR en la terminal)

Mismo WiFi:   USAR_HTTPS = True   (el celular y el PC en el mismo WiFi)
Sin WiFi:     USAR_HTTPS = False  (usa un túnel de Cloudflare, se descarga solo)

Teclas en la ventana del PC:  Q salir | V voz sí/no | M espejo
"""

import asyncio
import atexit
import datetime
import ipaddress
import os
import re
import shutil
import socket
import ssl
import subprocess
import threading
import time
import urllib.request

import cv2
from cvzone.FaceDetectionModule import FaceDetector
import numpy as np
from aiohttp import web, WSMsgType

# ═══════════════════════════════════════════════════════════
#  PARTE 1: SERVIDOR (recibe la cámara del celular)
# ═══════════════════════════════════════════════════════════
# ----- Configuración -----
PUERTO = 8443            # si está ocupado, cambia este número
# True  = mismo WiFi (el programa crea su propio https).
# False = usar con un TÚNEL (Cloudflare/ngrok) para funcionar SIN el mismo WiFi;
#         el túnel ya pone el https, así que aquí va en http normal.
USAR_HTTPS = False
ANCHO_ENVIO = 640        # ancho de la imagen que manda el celular (más chico = más rápido)
CALIDAD_JPEG = 0.6       # 0.1 (feo, rápido) a 1.0 (bonito, lento)
FPS_ENVIO = 20           # cuadros por segundo que intenta mandar el celular
CARPETA_CERT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "certificado")

# ----- Estado compartido -----
_frame_actual = None
_lock = threading.Lock()
_ultimo_frame_t = 0.0
_servidor_iniciado = False


def obtener_frame():
    """Devuelve el último frame (imagen BGR) que mandó el celular, o None."""
    with _lock:
        return None if _frame_actual is None else _frame_actual.copy()


def obtener_frame_con_tiempo(max_edad=3.0):
    """
    Devuelve (frame, instante). Si el celular no ha mandado nada en los
    últimos 'max_edad' segundos devuelve (None, 0.0). El 'instante' sirve
    para saber si el frame es NUEVO o es el mismo de antes.
    """
    with _lock:
        if _frame_actual is None or time.time() - _ultimo_frame_t > max_edad:
            return None, 0.0
        return _frame_actual.copy(), _ultimo_frame_t


# ---------------------------------------------------------------------------
# Página que se abre en el celular
# ---------------------------------------------------------------------------
HTML = """<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Cámara al computador</title>
<style>
  body { font-family: system-ui, sans-serif; background:#0d1117; color:#fff;
         display:flex; flex-direction:column; align-items:center;
         justify-content:center; min-height:100vh; margin:0; padding:20px;
         text-align:center; }
  button { font-size:20px; padding:16px 28px; margin:8px; border:0;
           border-radius:14px; background:#10a37f; color:#fff; }
  button.sec { background:#30363d; }
  #estado { margin-top:20px; font-size:18px; }
  video, canvas { display:none; }   /* la cámara NO se muestra en el celular */
</style>
</head>
<body>
  <h2>📱 → 💻</h2>
  <p>Toca el botón y acepta el permiso de la cámara.<br>
     La imagen se verá en el computador.</p>
  <button id="btnIniciar">Iniciar cámara</button>
  <button id="btnCambiar" class="sec" style="display:none">Cambiar cámara</button>
  <div id="estado">Listo para empezar</div>
  <video id="video" autoplay playsinline muted></video>
  <canvas id="canvas"></canvas>

<script>
const ANCHO = __ANCHO__, CALIDAD = __CALIDAD__, FPS = __FPS__;
const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const estado = document.getElementById("estado");
const btnIniciar = document.getElementById("btnIniciar");
const btnCambiar = document.getElementById("btnCambiar");

let ws = null, stream = null, timer = null, camara = "environment";

function conectarWS() {
  return new Promise((ok, fallo) => {
    ws = new WebSocket("wss://" + location.host + "/ws");
    ws.binaryType = "arraybuffer";
    ws.onopen = ok;
    ws.onerror = fallo;
    ws.onclose = () => { estado.textContent = "⚠️ Conexión cerrada"; detener(); };
  });
}

async function abrirCamara() {
  if (stream) stream.getTracks().forEach(t => t.stop());
  stream = await navigator.mediaDevices.getUserMedia({
    video: { facingMode: camara, width: { ideal: 1280 }, height: { ideal: 720 } },
    audio: false
  });
  video.srcObject = stream;
  await video.play();
}

function empezarEnvio() {
  clearInterval(timer);
  timer = setInterval(() => {
    if (!ws || ws.readyState !== 1 || !video.videoWidth) return;
    if (ws.bufferedAmount > 400000) return;          // evita atrasos
    const escala = ANCHO / video.videoWidth;
    canvas.width = ANCHO;
    canvas.height = Math.round(video.videoHeight * escala);
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    canvas.toBlob(b => { if (b && ws.readyState === 1) ws.send(b); },
                  "image/jpeg", CALIDAD);
  }, 1000 / FPS);
}

function detener() {
  clearInterval(timer);
  if (stream) stream.getTracks().forEach(t => t.stop());
}

btnIniciar.onclick = async () => {
  try {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      estado.textContent = "❌ Este link debe ser https (abre el que sale en el PC)";
      return;
    }
    estado.textContent = "Pidiendo permiso…";
    await abrirCamara();
    await conectarWS();
    empezarEnvio();
    btnIniciar.style.display = "none";
    btnCambiar.style.display = "inline-block";
    estado.textContent = "✅ Enviando la cámara al computador";
    try { await navigator.wakeLock.request("screen"); } catch (e) {}  // pantalla encendida
  } catch (e) {
    estado.textContent = "❌ " + e.name + ": " + e.message;
  }
};

btnCambiar.onclick = async () => {
  camara = (camara === "environment") ? "user" : "environment";
  try { await abrirCamara(); } catch (e) { estado.textContent = "❌ " + e.message; }
};
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Servidor (aiohttp)
# ---------------------------------------------------------------------------
async def _pagina(request):
    html = (HTML.replace("__ANCHO__", str(ANCHO_ENVIO))
                .replace("__CALIDAD__", str(CALIDAD_JPEG))
                .replace("__FPS__", str(FPS_ENVIO)))
    return web.Response(text=html, content_type="text/html")


async def _websocket(request):
    global _frame_actual, _ultimo_frame_t
    ws = web.WebSocketResponse(max_msg_size=4 * 1024 * 1024)
    await ws.prepare(request)
    print("📱 Celular conectado.")
    async for msg in ws:
        if msg.type == WSMsgType.BINARY:
            img = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)
            if img is not None:
                with _lock:
                    _frame_actual = img
                    _ultimo_frame_t = time.time()
        elif msg.type == WSMsgType.ERROR:
            break
    print("📱 Celular desconectado.")
    return ws


def _ip_local():
    """IP del computador dentro del WiFi."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))     # no manda nada, solo averigua la IP
        return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"
    finally:
        s.close()


def _crear_certificado(ip):
    """Crea (una vez por IP) un certificado propio para poder usar https."""
    from cryptography import x509
    from cryptography.x509.oid import NameOID
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa

    os.makedirs(CARPETA_CERT, exist_ok=True)
    ruta_cert = os.path.join(CARPETA_CERT, f"cert_{ip}.pem")
    ruta_key = os.path.join(CARPETA_CERT, f"key_{ip}.pem")
    if os.path.exists(ruta_cert) and os.path.exists(ruta_key):
        return ruta_cert, ruta_key

    llave = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    nombre = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, ip)])
    ahora = datetime.datetime.now(datetime.timezone.utc)
    cert = (x509.CertificateBuilder()
            .subject_name(nombre).issuer_name(nombre)
            .public_key(llave.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(ahora - datetime.timedelta(days=1))
            .not_valid_after(ahora + datetime.timedelta(days=365))
            .add_extension(x509.SubjectAlternativeName(
                [x509.IPAddress(ipaddress.ip_address(ip))]), critical=False)
            .sign(llave, hashes.SHA256()))

    with open(ruta_key, "wb") as f:
        f.write(llave.private_bytes(serialization.Encoding.PEM,
                                    serialization.PrivateFormat.TraditionalOpenSSL,
                                    serialization.NoEncryption()))
    with open(ruta_cert, "wb") as f:
        f.write(cert.public_bytes(serialization.Encoding.PEM))
    return ruta_cert, ruta_key


def _correr_servidor(ip, ruta_cert, ruta_key):
    async def main():
        app = web.Application()
        app.router.add_get("/", _pagina)
        app.router.add_get("/ws", _websocket)
        contexto = None
        if USAR_HTTPS:
            contexto = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            contexto.load_cert_chain(ruta_cert, ruta_key)
        runner = web.AppRunner(app)
        await runner.setup()
        host = "0.0.0.0" if USAR_HTTPS else "127.0.0.1"   # con túnel basta localhost
        await web.TCPSite(runner, host, PUERTO, ssl_context=contexto).start()
        await asyncio.Event().wait()      # quedarse corriendo para siempre

    asyncio.run(main())


def iniciar_servidor():
    """Arranca el servidor en un hilo y devuelve el link para el celular."""
    global _servidor_iniciado
    ip = _ip_local()
    link = f"https://{ip}:{PUERTO}" if USAR_HTTPS else f"http://localhost:{PUERTO}"
    if _servidor_iniciado:
        return link
    ruta_cert, ruta_key = (_crear_certificado(ip) if USAR_HTTPS else (None, None))
    threading.Thread(target=_correr_servidor, args=(ip, ruta_cert, ruta_key),
                     daemon=True).start()
    _servidor_iniciado = True
    return link


def _imprimir_qr(link):
    try:
        import qrcode
        qr = qrcode.QRCode(border=1)
        qr.add_data(link)
        qr.print_ascii(invert=True)
    except Exception:
        pass


URL_CLOUDFLARED_WINDOWS = ("https://github.com/cloudflare/cloudflared/releases/"
                           "latest/download/cloudflared-windows-amd64.exe")


def _buscar_cloudflared():
    """Devuelve la ruta de cloudflared; en Windows lo descarga solo si no está."""
    ruta = shutil.which("cloudflared")
    if ruta:
        return ruta
    aqui = os.path.dirname(os.path.abspath(__file__))
    local = os.path.join(aqui, "cloudflared.exe")
    if os.path.exists(local):
        return local
    if os.name == "nt":
        print("\n⬇️  Descargando cloudflared (solo la primera vez, ~40 MB)...")
        try:
            urllib.request.urlretrieve(URL_CLOUDFLARED_WINDOWS, local)
            print("✅ Descargado.")
            return local
        except Exception as e:
            print(f"❌ No pude descargarlo: {e}")
    return None


def _iniciar_tunel():
    """Lanza cloudflared solo y muestra el link público (https://...trycloudflare.com)."""
    ruta = _buscar_cloudflared()
    if not ruta:
        print("\n❌ No tengo cloudflared. Descárgalo desde:")
        print("     https://github.com/cloudflare/cloudflared/releases/latest")
        print("   (cloudflared-windows-amd64.exe), ponlo en la misma carpeta que este")
        print("   programa con el nombre cloudflared.exe y corre esto de nuevo.\n")
        return
    try:
        proc = subprocess.Popen(
            [ruta, "tunnel", "--url", f"http://localhost:{PUERTO}"],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="ignore")
    except Exception as e:
        print(f"\n❌ No pude iniciar cloudflared: {e}\n")
        return
    atexit.register(proc.terminate)
    print("\n⏳ Creando el túnel, espera unos segundos...")

    def leer():
        mostrado = False
        for linea in proc.stdout:
            m = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com", linea)
            if m and not mostrado:
                mostrado = True
                link = m.group(0)
                print("\n" + "=" * 56)
                print("  Abre este link en CHROME del celular (datos o cualquier WiFi):\n")
                print(f"      {link}\n")
                print("=" * 56 + "\n")
                _imprimir_qr(link)
        print("\n[El túnel se cerró. Si no saliste tú, revisa tu internet.]")

    threading.Thread(target=leer, daemon=True).start()


def _mostrar_link(link):
    print("\n" + "=" * 56)
    print("  Abre este link en CHROME del celular (mismo WiFi):\n")
    print(f"      {link}\n")
    print("  Chrome dirá 'conexión no privada': toca")
    print("  'Configuración avanzada' -> 'Acceder a ... (no seguro)'.")
    print("=" * 56 + "\n")
    try:
        import qrcode
        qr = qrcode.QRCode(border=1)
        qr.add_data(link)
        qr.print_ascii(invert=True)
    except Exception:
        print("(Tip: pip install qrcode  para que salga un QR y no tengas que escribir el link)")


def mostrar_acceso():
    """Arranca el servidor y muestra cómo entrar desde el celular (link/QR/túnel)."""
    link = iniciar_servidor()
    if USAR_HTTPS:
        _mostrar_link(link)
    else:
        _iniciar_tunel()
    return link


# ═══════════════════════════════════════════════════════════
#  PARTE 2: DETECCIÓN (cuenta cuántas personas hay)
# ═══════════════════════════════════════════════════════════

# La voz es opcional: si no está pyttsx3, el mensaje igual sale en pantalla.
try:
    import pyttsx3
    HAY_VOZ = True
except Exception:
    HAY_VOZ = False

detector_rostro = FaceDetector(minDetectionCon=0.7)

# Esto arranca el servidor y muestra el link / QR / túnel para el celular.
mostrar_acceso()
print("✅ Esperando al celular...  Q = salir, V = voz sí/no, M = espejo.")

# El número solo cambia si se mantiene igual varios fotogramas seguidos
# (así no salta por un parpadeo de la detección). Si lo sientes lento, bájalo a 5.
FRAMES_ESTABLES = 10
personas_confirmadas = None   # None = todavía no sabe
candidato = None
frames_candidato = 0
voz_activa = True
espejo = False                # cámara trasera = sin espejo; frontal = tecla M
ultimo_t_frame = 0.0
TITULO = "Personas  |  V=voz  M=espejo  Q=salir"


def texto_personas(n):
    if n is None:
        return "Buscando..."
    if n == 0:
        return "No hay nadie"
    if n == 1:
        return "Hay una persona"
    return f"Hay {n} personas"


def decir(texto):
    """Habla en un hilo aparte para no congelar el video."""
    if not (HAY_VOZ and voz_activa):
        return

    def _hablar():
        try:
            motor = pyttsx3.init()
            motor.setProperty("rate", 175)
            for voz in motor.getProperty("voices"):
                etiqueta = f"{voz.name} {voz.id}".lower()
                if any(p in etiqueta for p in ("spanish", "español", "es-", "es_",
                                               "helena", "sabina", "mónica", "monica")):
                    motor.setProperty("voice", voz.id)
                    break
            motor.say(texto)
            motor.runAndWait()
            motor.stop()
        except Exception as e:
            print(f"[No pude hablar: {e}]")

    threading.Thread(target=_hablar, daemon=True).start()


def actualizar_conteo(detectadas):
    """Confirma el nuevo número solo si dura FRAMES_ESTABLES fotogramas."""
    global personas_confirmadas, candidato, frames_candidato
    if detectadas == personas_confirmadas:
        candidato, frames_candidato = detectadas, 0
        return
    if detectadas == candidato:
        frames_candidato += 1
    else:
        candidato, frames_candidato = detectadas, 1
    if frames_candidato >= FRAMES_ESTABLES:
        personas_confirmadas = detectadas
        frames_candidato = 0
        mensaje = texto_personas(personas_confirmadas)
        print(f"👥 {mensaje}")
        decir(mensaje)


def dibujar_contador(frame, n):
    """Cartel grande arriba a la izquierda con el mensaje."""
    texto = texto_personas(n)
    (tw, th), _ = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
    color = (0, 160, 0) if n else (90, 90, 90)
    cv2.rectangle(frame, (10, 10), (30 + tw, 30 + th + 10), color, -1)
    cv2.putText(frame, texto, (20, 25 + th), cv2.FONT_HERSHEY_SIMPLEX,
                1.2, (255, 255, 255), 3, cv2.LINE_AA)


def leer_teclas(espera_ms):
    """Atiende las teclas. Devuelve True si hay que salir."""
    global voz_activa, espejo
    tecla = cv2.waitKey(espera_ms) & 0xFF
    if tecla == ord('q'):
        return True
    elif tecla == ord('v'):
        voz_activa = not voz_activa
        print("🔊 Voz activada" if voz_activa else "🔇 Voz silenciada")
    elif tecla == ord('m'):
        espejo = not espejo
    return False


# ═══════════════════════════════════
#  BUCLE PRINCIPAL
# ═══════════════════════════════════
while True:
    # Tomar el último frame que mandó el celular
    frame, t_frame = obtener_frame_con_tiempo(max_edad=3.0)

    if frame is None:                      # el celular aún no manda nada
        espera = np.zeros((360, 640, 3), dtype=np.uint8)
        cv2.putText(espera, "Esperando al celular...", (130, 190),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow(TITULO, espera)
        if leer_teclas(30):
            break
        continue

    if t_frame == ultimo_t_frame:          # mismo frame de antes: no repetir el trabajo
        if leer_teclas(5):
            break
        continue
    ultimo_t_frame = t_frame

    if espejo:
        frame = cv2.flip(frame, 1)

    # Contar rostros = contar personas
    frame, caras = detector_rostro.findFaces(frame, draw=False)
    actualizar_conteo(len(caras) if caras else 0)

    dibujar_contador(frame, personas_confirmadas)
    cv2.imshow(TITULO, frame)

    if leer_teclas(1):
        break

cv2.destroyAllWindows()
print("✅ Programa cerrado correctamente.")
