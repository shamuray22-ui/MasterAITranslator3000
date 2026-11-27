import re
import time
import threading
import unicodedata
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


def _caixa_delimitadora(caixa):
    """Retorna x1, y1, x2, y2 para as caixas quadrilaterais do RapidOCR."""
    pontos = np.asarray(caixa, dtype=float).reshape(-1, 2)
    return (
        float(pontos[:, 0].min()), float(pontos[:, 1].min()),
        float(pontos[:, 0].max()), float(pontos[:, 1].max()),
    )


def _deteccoes_sobrepostas(caixa_a, caixa_b):
    ax1, ay1, ax2, ay2 = caixa_a
    bx1, by1, bx2, by2 = caixa_b
    largura = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    altura = max(0.0, min(ay2, by2) - max(ay1, by1))
    intersecao = largura * altura
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    uniao = area_a + area_b - intersecao
    return uniao > 0 and intersecao / uniao >= 0.5


def _chave_texto(texto):
    texto = unicodedata.normalize("NFKC", texto).casefold()
    return re.sub(r"\W+", "", texto, flags=re.UNICODE)


def _extrair_textos_resultado(resultado):
    """Extrai, ordena e remove deteccoes duplicadas/sobrepostas do RapidOCR."""
    deteccoes = []

    if hasattr(resultado, "txts"):
        caixas = getattr(resultado, "boxes", None)
        for indice, texto in enumerate(getattr(resultado, "txts", [])):
            texto_limpo = str(texto).strip()
            if len(texto_limpo) <= 1:
                continue
            caixa = caixas[indice] if caixas is not None and indice < len(caixas) else None
            try:
                limites = _caixa_delimitadora(caixa) if caixa is not None else None
            except (TypeError, ValueError):
                limites = None
            deteccoes.append((limites, texto_limpo))

    elif resultado:
        for linha in resultado:
            try:
                texto_limpo = str(linha[1]).strip()
                if len(texto_limpo) <= 1:
                    continue
                limites = _caixa_delimitadora(linha[0])
                deteccoes.append((limites, texto_limpo))
            except (IndexError, TypeError, ValueError):
                continue

    deteccoes.sort(key=lambda item: (
        item[0][1] if item[0] is not None else 0,
        item[0][0] if item[0] is not None else 0,
    ))

    unicas = []
    for limites, texto in deteccoes:
        chave = _chave_texto(texto)
        duplicada = any(
            chave == _chave_texto(texto_existente)
            and limites is not None
            and limites_existentes is not None
            and _deteccoes_sobrepostas(limites, limites_existentes)
            for limites_existentes, texto_existente in unicas
        )
        if not duplicada:
            unicas.append((limites, texto))

    return [texto for _, texto in unicas]


def _preprocessar_imagem(imagem):
    """Amplia e melhora o contraste sem binarizar texto sobre fundos variados."""
    imagem = cv2.cvtColor(imagem, cv2.COLOR_BGRA2BGR)
    imagem = cv2.resize(imagem, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    imagem_cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    imagem_contraste = clahe.apply(imagem_cinza)
    return cv2.cvtColor(imagem_contraste, cv2.COLOR_GRAY2BGR)


def executar_ocr():
    global texto_original_global, traducao_atual_global
    global posicao_traducao_global, ultimo_tempo_traducao
    texto_anterior = ""

    try:
        with mss.mss() as sct:
            # monitor 0 e a area virtual completa, inclusive monitores com origem negativa.
            area_tela = sct.monitors[0]

            while executando:
                try:
                    mx, my = pyautogui.position()
                    metade = TAMANHO_CAIXA_MOUSE // 2
                    esquerda = max(area_tela["left"], mx - metade)
                    topo = max(area_tela["top"], my - metade)
                    direita = min(
                        area_tela["left"] + area_tela["width"], mx + metade
                    )
                    base = min(
                        area_tela["top"] + area_tela["height"], my + metade
                    )

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
