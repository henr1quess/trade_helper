import time
import random
import datetime
import sys
import platform
from pynput.keyboard import Controller

kb = Controller()

def beep():
    try:
        if platform.system() == "Windows":
            import winsound
            winsound.Beep(1000, 200)  # 200 ms
        else:
            sys.stdout.write('\a'); sys.stdout.flush()
    except Exception:
        pass

def tap_key(char: str, seconds: float):
    kb.press(char)
    time.sleep(seconds)
    kb.release(char)

def format_ms(s):
    return f"{int(s*1000)}ms"

print("Iniciando lembretes: intervalo aleatório entre 4 e 10 minutos.")
print("Quando avisar, vou acionar A e D em ordem aleatória (~50/50). Ctrl+C para encerrar.")

try:
    while True:
        intervalo = random.uniform(4*60, 10*60)  # 4–10 min
        proximo = datetime.datetime.now() + datetime.timedelta(seconds=intervalo)
        print(f"\n[{datetime.datetime.now():%H:%M:%S}] Próximo aviso às {proximo:%H:%M:%S} "
              f"(em {intervalo/60:.1f} min)")
        time.sleep(intervalo)

        # Durações “humanas” independentes para A e D (30–120 ms)
        dur = {
            'a': random.uniform(0.030, 0.120),
            'd': random.uniform(0.030, 0.120),
        }
        # Delay “humano” entre as teclas (40–160 ms)
        delay_entre = random.uniform(0.040, 0.160)

        # Ordem aleatória com probabilidade 50/50
        first, second = random.choice([('a','d'), ('d','a')])

        beep()
        agora = datetime.datetime.now().strftime("%H:%M:%S")
        print(f"[{agora}] Ordem: {first.upper()} depois {second.upper()}")
        print(f"  • {first.upper()} (toque ~ {format_ms(dur[first])})")
        print(f"  • esperar ~ {format_ms(delay_entre)}")
        print(f"  • {second.upper()} (toque ~ {format_ms(dur[second])})")

        # Executa
        tap_key(first, dur[first])
        time.sleep(delay_entre)
        tap_key(second, dur[second])

except KeyboardInterrupt:
    print("\nEncerrado pelo usuário.")
