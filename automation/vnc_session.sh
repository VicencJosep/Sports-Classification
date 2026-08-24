#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# automation/vnc_session.sh
# ===========================
# Levanta un display virtual (Xvfb) + servidor VNC dentro del contenedor, para
# poder VER e INTERACTUAR con el Chromium que lanza automation/molmo_publisher.py.
#
# Existe porque este host es headless: no hay DISPLAY, no hay sockets en
# /tmp/.X11-unix y la sesión es tty. Sin esto, `headless=False` no tiene dónde
# dibujar y el login manual de Google (que NO se automatiza: 2FA, CAPTCHAs) es
# imposible.
#
# Uso (dentro del contenedor):
#     bash automation/vnc_session.sh start      # arranca Xvfb + x11vnc
#     bash automation/vnc_session.sh status
#     bash automation/vnc_session.sh stop
#
# Desde tu portátil, túnel SSH contra la IP del contenedor en la red bridge
# (no hace falta publicar puertos ni reiniciar el contenedor):
#     ssh -L 5900:172.17.0.3:5900 <usuario>@<este-host>
# y conecta tu cliente VNC a localhost:5900.
#
# La contraseña se genera sola la primera vez y se imprime por pantalla. Vive
# en /root/.vnc/passwd DENTRO del contenedor (no en el repo montado), así que
# no se versiona ni sobrevive a un `docker rm`.
# ---------------------------------------------------------------------------
set -euo pipefail

DISPLAY_NUM="${DISPLAY_NUM:-99}"
SCREEN="${SCREEN:-1440x900x24}"   # debe cuadrar con config.PublisherConfig.viewport_*
VNC_PORT="${VNC_PORT:-5900}"
PASSWD_FILE="/root/.vnc/passwd"

# ¿Hay un proceso VIVO que case con el patrón? Los zombis no cuentan.
#
# Importa aquí y no es paranoia: el PID 1 de este contenedor es
# `tail -f /dev/null`, que no recolecta hijos huérfanos. Cuando x11vnc muere
# deja una entrada <defunct> que `pgrep` sigue listando, y sin este filtro el
# script se cree arrancado mientras el puerto 5900 no lo escucha nadie.
alive() {
    local pid
    for pid in $(pgrep -f "$1" 2>/dev/null); do
        [ "$pid" = "$$" ] && continue
        case "$(ps -o stat= -p "$pid" 2>/dev/null | tr -d ' ')" in
            Z*) ;;            # zombi: ignóralo
            "") ;;            # ya no existe
            *) return 0 ;;    # vivo de verdad
        esac
    done
    return 1
}

start() {
    if alive "[X]vfb :${DISPLAY_NUM}" && [ -e "/tmp/.X${DISPLAY_NUM}-lock" ]; then
        echo "Xvfb ya corría en :${DISPLAY_NUM}"
    else
        # nohup + redirección: sin esto Xvfb retiene stdout y el `docker exec`
        # que llama a este script no vuelve nunca.
        nohup Xvfb ":${DISPLAY_NUM}" -screen 0 "${SCREEN}" -nolisten tcp \
            >/tmp/xvfb.log 2>&1 &
        disown || true
        sleep 2
        echo "Xvfb arrancado en :${DISPLAY_NUM} (${SCREEN})"
    fi

    if [ ! -f "${PASSWD_FILE}" ]; then
        mkdir -p "$(dirname "${PASSWD_FILE}")"
        # Contraseña de un solo uso: el VNC queda accesible desde la red bridge
        # de Docker (host + resto de contenedores), así que sin auth no se deja.
        pass="$(head -c 6 /dev/urandom | base64 | tr -dc 'A-Za-z0-9' | head -c 8)"
        x11vnc -storepasswd "${pass}" "${PASSWD_FILE}" >/dev/null 2>&1
        echo "-----------------------------------------------"
        echo "  Contraseña VNC (generada ahora): ${pass}"
        echo "  Guardada en ${PASSWD_FILE}"
        echo "-----------------------------------------------"
    else
        echo "Reutilizando la contraseña ya guardada en ${PASSWD_FILE}"
    fi

    if alive "[x]11vnc -display"; then
        echo "x11vnc ya corría"
    else
        rm -f "/tmp/.X${DISPLAY_NUM}-lock.vnc" 2>/dev/null || true
        x11vnc -display ":${DISPLAY_NUM}" -rfbauth "${PASSWD_FILE}" \
               -rfbport "${VNC_PORT}" -forever -shared -noxdamage -quiet \
               -bg >/dev/null
        echo "x11vnc escuchando en :${VNC_PORT}"
    fi

    ip="$(hostname -i 2>/dev/null | awk '{print $1}')"
    cat <<EOF

Listo. Desde tu portátil:
    ssh -L ${VNC_PORT}:${ip}:${VNC_PORT} <usuario>@<este-host>
    # y conecta el cliente VNC a localhost:${VNC_PORT}

Después, dentro del contenedor:
    export DISPLAY=:${DISPLAY_NUM}
    python -m automation.molmo_publisher --login-only
EOF
}

stop() {
    # Patrones entre corchetes para que pkill -f no case con la propia línea
    # de comandos de este script y se mate a sí mismo.
    pkill -f "[x]11vnc -display" 2>/dev/null && echo "x11vnc parado" || echo "x11vnc no estaba corriendo"
    pkill -f "[X]vfb :${DISPLAY_NUM}" 2>/dev/null && echo "Xvfb parado" || echo "Xvfb no estaba corriendo"
    rm -f "/tmp/.X${DISPLAY_NUM}-lock"
}

status() {
    alive "[X]vfb :${DISPLAY_NUM}" && echo "Xvfb  :${DISPLAY_NUM} OK" || echo "Xvfb  parado"
    alive "[x]11vnc -display" && echo "x11vnc proceso OK" || echo "x11vnc parado"
    # La prueba de verdad no es el proceso, es que alguien escuche el puerto.
    python3 - <<PY
import socket
s = socket.socket(); s.settimeout(2)
try:
    s.connect(("127.0.0.1", ${VNC_PORT})); print("puerto ${VNC_PORT} ACEPTA conexiones")
except Exception as e:
    print("puerto ${VNC_PORT} NO responde:", e)
finally:
    s.close()
PY
}

case "${1:-start}" in
    start)  start ;;
    stop)   stop ;;
    status) status ;;
    *) echo "uso: $0 {start|stop|status}"; exit 1 ;;
esac
