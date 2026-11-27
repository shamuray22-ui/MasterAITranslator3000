import time
import threading
from pathlib import Path

import argostranslate.package
import argostranslate.translate
import cv2
import mss
import numpy as np
import pyautogui
from rapidocr_onnxruntime import RapidOCR


IDIOMA_ORIGEM = "en"
IDIOMA_DESTINO = "pt"
TAMANHO_CAIXA_MOUSE = 200
TEMPO_TOLERANCIA_DETECCAO = 1.2

engine = None
executando = False
texto_original_global = ""
traducao_atual_global = ""
posicao_traducao_global = (0, 0)
ultimo_tempo_traducao = 0


def iniciar_tradutor(modelos):
    """Valida os modelos, prepara o OCR/traducao e inicia o motor."""
    global engine, executando

    if bool(modelos["det"]) != bool(modelos["rec"]):
        raise ValueError(
            "Deteccao e reconhecimento precisam ser selecionados juntos."
        )

    for tipo, caminho in modelos.items():
        if caminho and not Path(caminho).is_file():
            raise FileNotFoundError(
                f"O arquivo selecionado nao existe ({tipo}):\n{caminho}"
            )

    parametros = {}
    if modelos["det"]:
        parametros["det_model_path"] = modelos["det"]
        parametros["rec_model_path"] = modelos["rec"]
    if modelos["cls"]:
        parametros["cls_model_path"] = modelos["cls"]
    if modelos["keys"]:
        parametros["rec_keys_path"] = modelos["keys"]

    print("[DEBUG] Inicializando OCR...")
    engine = RapidOCR(**parametros)
    print("[DEBUG] OCR carregado!")

    traducao_instalada = _obter_traducao_instalada()
    if not traducao_instalada and modelos["argos"]:
        print("[DEBUG] Instalando pacote Argos...")
        argostranslate.package.install_from_path(modelos["argos"])
        traducao_instalada = _obter_traducao_instalada()

    if not traducao_instalada:
        raise RuntimeError(
            "Traducao en -> pt nao encontrada.\n"
            "Selecione um pacote Argos ou instale-o previamente."
        )

    executando = True
    threading.Thread(target=executar_ocr, daemon=True).start()
    print("[PRONTO] Tradutor iniciado!")


def _obter_traducao_instalada():
    idiomas = argostranslate.translate.get_installed_languages()
    origem = next((idioma for idioma in idiomas if idioma.code == IDIOMA_ORIGEM), None)
    destino = next((idioma for idioma in idiomas if idioma.code == IDIOMA_DESTINO), None)
    return origem.get_translation(destino) if origem and destino else None


def parar_tradutor():
    global executando
    executando = False


def executar_ocr():
    global texto_original_global, traducao_atual_global
    global posicao_traducao_global, ultimo_tempo_traducao
    texto_anterior = ""

    try:
        with mss.mss() as sct:
            while executando:
                try:
                    mx, my = pyautogui.position()
                    metade = TAMANHO_CAIXA_MOUSE // 2
                    monitor = {
                        "top": max(0, my - metade),
                        "left": max(0, mx - metade),
                        "width": TAMANHO_CAIXA_MOUSE,
                        "height": TAMANHO_CAIXA_MOUSE,
                    }
                    imagem = np.array(sct.grab(monitor))
                    imagem = cv2.cvtColor(imagem, cv2.COLOR_BGRA2BGR)
                    resposta = engine(imagem)
                    resultado = resposta[0] if isinstance(resposta, tuple) else resposta

                    if hasattr(resultado, "txts"):
                        textos = [
                            texto for texto in resultado.txts
                            if texto and len(texto.strip()) > 1
                        ]
                    elif resultado:
                        textos = [
                            linha[1] for linha in resultado
                            if len(linha[1].strip()) > 1
                        ]
                    else:
                        textos = []

                    if textos:
                        texto = " ".join(textos).strip()
                        agora = time.time()
                        posicao_traducao_global = (mx, my)
                        ultimo_tempo_traducao = agora
                        texto_original_global = texto

                        # Traduz novamente apenas quando o texto reconhecido mudar.
                        if texto != texto_anterior:
                            texto_anterior = texto
                            traducao = argostranslate.translate.translate(
                                texto, IDIOMA_ORIGEM, IDIOMA_DESTINO
                            )
                            traducao_atual_global = traducao
                            print(f"[TRADUZIDO]: {texto} -> {traducao}")
                    elif time.time() - ultimo_tempo_traducao > TEMPO_TOLERANCIA_DETECCAO:
                        # Uma pequena tolerancia evita piscadas causadas por falhas
                        # momentaneas do OCR; depois disso, o overlay volta ao modo
                        # de busca e a mesma frase pode ser traduzida novamente.
                        texto_anterior = ""
                        texto_original_global = ""
                        traducao_atual_global = ""
                except Exception as erro:
                    print(f"[ERRO OCR]: {type(erro).__name__}: {erro}")
                time.sleep(0.8)
    except Exception as erro:
        print(f"[ERRO MOTOR]: {erro}")
        parar_tradutor()


if __name__ == "__main__":
    print("Execute InterfaceMilionarie.py para abrir a interface do tradutor.")
