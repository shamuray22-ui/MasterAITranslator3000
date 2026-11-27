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


def _extrair_textos_resultado(resultado):
    """Extrai textos do RapidOCR suportando resultados antigos e novos."""
    linhas = []

    if hasattr(resultado, "txts"):
        caixas = getattr(resultado, "boxes", None)
        txts = getattr(resultado, "txts", [])
        for indice, texto in enumerate(txts):
            texto_limpo = str(texto).strip()
            if len(texto_limpo) <= 1:
                continue

            caixa = caixas[indice] if caixas is not None and indice < len(caixas) else None
            if caixa is not None:
                ponto_superior = min(float(ponto[1]) for ponto in caixa)
                ponto_esquerdo = min(float(ponto[0]) for ponto in caixa)
            else:
                ponto_superior = len(linhas)
                ponto_esquerdo = 0
            linhas.append((ponto_superior, ponto_esquerdo, texto_limpo))

    elif resultado:
        for linha in resultado:
            try:
                caixa = linha[0]
                texto_limpo = str(linha[1]).strip()
                if len(texto_limpo) <= 1:
                    continue
                ponto_superior = min(float(ponto[1]) for ponto in caixa)
                ponto_esquerdo = min(float(ponto[0]) for ponto in caixa)
                linhas.append((ponto_superior, ponto_esquerdo, texto_limpo))
            except (IndexError, TypeError, ValueError):
                continue

    linhas.sort(key=lambda item: (item[0], item[1]))
    return [item[2] for item in linhas]


def _preprocessar_imagem(imagem):
    """Melhora contraste e nitidez da pequena area capturada pelo OCR."""
    imagem = cv2.cvtColor(imagem, cv2.COLOR_BGRA2BGR)
    imagem = cv2.resize(imagem, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    imagem_cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    imagem_cinza = cv2.bilateralFilter(imagem_cinza, 7, 50, 50)
    _, imagem_escura = cv2.threshold(imagem_cinza, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2.cvtColor(imagem_escura, cv2.COLOR_GRAY2BGR)


def executar_ocr():
    global texto_original_global, traducao_atual_global
    global posicao_traducao_global, ultimo_tempo_traducao
    texto_anterior = ""

    try:
        with mss.mss() as sct:
            monitor_principal = sct.monitors[1]
            largura_tela = monitor_principal["width"]
            altura_tela = monitor_principal["height"]

            while executando:
                try:
                    mx, my = pyautogui.position()
                    metade = TAMANHO_CAIXA_MOUSE // 2
                    esquerda = max(0, mx - metade)
                    topo = max(0, my - metade)
                    direita = min(largura_tela, mx + metade)
                    base = min(altura_tela, my + metade)

                    monitor = {
                        "top": topo,
                        "left": esquerda,
                        "width": max(1, direita - esquerda),
                        "height": max(1, base - topo),
                    }
                    imagem = np.array(sct.grab(monitor))
                    imagem = _preprocessar_imagem(imagem)
                    resposta = engine(imagem)
                    resultado = resposta[0] if isinstance(resposta, tuple) else resposta
                    textos = _extrair_textos_resultado(resultado)

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
