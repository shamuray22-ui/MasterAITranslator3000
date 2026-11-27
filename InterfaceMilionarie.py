import json
import threading
import time
import tkinter as tk
import tkinter.font as tkfont
import urllib.parse
import urllib.request
from pathlib import Path
from tkinter import filedialog, messagebox
from tkinter import ttk

import run


ARQUIVO_CONFIG = Path(__file__).with_name("modelos.json")
modelos = {"det": "", "rec": "", "cls": "", "keys": "", "argos": ""}
campos = {}
resultados_huggingface = []
categoria_huggingface = "det"

CATEGORIAS_HUGGINGFACE = {
    "det": {"nome": "Deteccao OCR", "busca": "OCR detection ONNX", "extensoes": (".onnx",)},
    "rec": {"nome": "Reconhecimento OCR", "busca": "OCR recognition ONNX", "extensoes": (".onnx",)},
    "cls": {"nome": "Classificacao OCR", "busca": "OCR classification ONNX", "extensoes": (".onnx",)},
    "keys": {"nome": "Dicionario OCR", "busca": "OCR dictionary", "extensoes": (".txt",)},
    "argos": {"nome": "Traducao Argos", "busca": "Argos translate en pt", "extensoes": (".argosmodel",)},
}


def obter_arquivos_compativeis(modelo, categoria):
    extensoes = CATEGORIAS_HUGGINGFACE[categoria]["extensoes"]
    arquivos = [
        item.get("rfilename", "")
        for item in modelo.get("siblings", [])
        if item.get("rfilename", "").lower().endswith(extensoes)
    ]
    return arquivos


def buscar_modelos_huggingface(categoria, termo):
    limite = 30
    parametros = urllib.parse.urlencode({
        "search": termo,
        "sort": "downloads",
        "direction": "-1",
        "limit": limite,
        "full": "true",
    })
    url = f"https://huggingface.co/api/models?{parametros}"
    janela_config.after(0, lambda: atualizar_status_huggingface(
        "Consultando Hugging Face..."
    ))

    try:
        with urllib.request.urlopen(url, timeout=20) as resposta:
            modelos_encontrados = json.loads(resposta.read().decode("utf-8"))

        resultados = []
        for modelo in modelos_encontrados:
            arquivos = obter_arquivos_compativeis(modelo, categoria)
            if arquivos:
                resultados.append({
                    "repo_id": modelo.get("id", ""),
                    "arquivo": arquivos[0],
                    "arquivos": arquivos,
                    "downloads": modelo.get("downloads", 0),
                })

        janela_config.after(0, lambda: mostrar_resultados_huggingface(resultados))
    except Exception as erro:
        janela_config.after(0, lambda: atualizar_status_huggingface(
            f"Erro na busca: {erro}"
        ))


def mostrar_resultados_huggingface(resultados):
    global resultados_huggingface
    resultados_huggingface = resultados
    lista_huggingface.delete(0, tk.END)
    for resultado in resultados:
        lista_huggingface.insert(
            tk.END,
            f"{resultado['repo_id']}  |  {resultado['arquivo']}  |  "
            f"downloads: {resultado['downloads']}",
        )
    atualizar_status_huggingface(f"{len(resultados)} modelo(s) encontrado(s).")


def atualizar_status_huggingface(texto):
    status_huggingface.set(texto)


def selecionar_resultado_huggingface(_evento=None):
    selecionados = lista_huggingface.curselection()
    if not selecionados:
        return
    resultado = resultados_huggingface[selecionados[0]]
    arquivo_selecionado_huggingface.set(resultado["arquivo"])
    if len(resultado["arquivos"]) > 1:
        atualizar_status_huggingface(
            "Este repositorio possui varios arquivos compativeis; "
            "sera usado o primeiro."
        )


def baixar_modelo_huggingface():
    selecionados = lista_huggingface.curselection()
    if not selecionados:
        messagebox.showwarning("Nenhum modelo", "Selecione um modelo na lista.")
        return

    resultado = resultados_huggingface[selecionados[0]]
    arquivo = arquivo_selecionado_huggingface.get() or resultado["arquivo"]
    destino = (
        Path(__file__).with_name("modelos_huggingface")
        / categoria_huggingface
        / resultado["repo_id"].replace("/", "_")
        / Path(arquivo).name
    )
    destino.parent.mkdir(parents=True, exist_ok=True)
    download_modelo_huggingface(resultado["repo_id"], arquivo, destino)


def iniciar_busca_huggingface():
    categoria = categoria_huggingface
    termo = busca_huggingface.get().strip()
    threading.Thread(
        target=buscar_modelos_huggingface,
        args=(categoria, termo),
        daemon=True,
    ).start()


def download_modelo_huggingface(repo_id, arquivo, destino):
    def baixar():
        try:
            nome = urllib.parse.quote(arquivo, safe="/")
            url = f"https://huggingface.co/{repo_id}/resolve/main/{nome}"
            janela_config.after(0, lambda: atualizar_status_huggingface(
                f"Baixando {arquivo}..."
            ))
            urllib.request.urlretrieve(url, destino)
            janela_config.after(0, lambda: concluir_download_huggingface(destino))
        except Exception as erro:
            janela_config.after(0, lambda: atualizar_status_huggingface(
                f"Erro no download: {erro}"
            ))

    threading.Thread(target=baixar, daemon=True).start()


def concluir_download_huggingface(destino):
    campo = campos[categoria_huggingface]
    campo.delete(0, tk.END)
    campo.insert(0, destino.as_posix())
    salvar_config()
    atualizar_status_huggingface(f"Baixado e selecionado: {destino.name}")


def salvar_config():
    try:
        for tipo, campo in campos.items():
            modelos[tipo] = campo.get().strip().replace("\\", "/")
        ARQUIVO_CONFIG.write_text(
            json.dumps(modelos, indent=4, ensure_ascii=False), encoding="utf-8"
        )
        print("[CONFIG] Caminhos salvos!")
    except Exception as erro:
        print(f"[CONFIG] Erro ao salvar: {erro}")


def carregar_config():
    if not ARQUIVO_CONFIG.exists():
        return
    try:
        dados = json.loads(ARQUIVO_CONFIG.read_text(encoding="utf-8"))
        for tipo in modelos:
            caminho = dados.get(tipo, "")
            if isinstance(caminho, str):
                modelos[tipo] = caminho
                if tipo in campos:
                    campos[tipo].insert(0, caminho)
        print("[CONFIG] Caminhos recuperados!")
    except Exception as erro:
        print(f"[CONFIG] Erro ao carregar: {erro}")


def selecionar_arquivo(tipo):
    filtros = {
        "det": [("Modelo ONNX", "*.onnx")],
        "rec": [("Modelo ONNX", "*.onnx")],
        "cls": [("Modelo ONNX", "*.onnx")],
        "keys": [("Dicionario", "*.txt")],
        "argos": [("Modelo Argos", "*.argosmodel")],
    }
    arquivo = filedialog.askopenfilename(
        title="Selecionar modelo", filetypes=filtros[tipo]
    )
    if arquivo:
        campos[tipo].delete(0, tk.END)
        campos[tipo].insert(0, arquivo)
        salvar_config()


def iniciar_programa():
    salvar_config()
    try:
        run.iniciar_tradutor(modelos)
        iniciar_overlay()
    except Exception as erro:
        run.parar_tradutor()
        messagebox.showerror("Erro ao iniciar", f"{type(erro).__name__}:\n{erro}")
        print(f"[ERRO COMPLETO] {type(erro).__name__}: {erro}")


def quebrar_linhas(texto, fonte, largura_maxima):
    """Quebra texto por palavras respeitando a largura medida pela fonte."""
    linhas = []
    linha_atual = ""
    for palavra in texto.split():
        candidato = f"{linha_atual} {palavra}".strip()
        if linha_atual and fonte.measure(candidato) > largura_maxima:
            linhas.append(linha_atual)
            linha_atual = palavra
        else:
            linha_atual = candidato
    if linha_atual:
        linhas.append(linha_atual)
    return "\n".join(linhas) or " "


def pontos_retangulo_arredondado(x1, y1, x2, y2, raio=12):
    raio = min(raio, (x2 - x1) / 2, (y2 - y1) / 2)
    return [
        x1 + raio, y1, x2 - raio, y1,
        x2, y1, x2, y1 + raio,
        x2, y2 - raio, x2, y2,
        x2 - raio, y2, x1 + raio, y2,
        x1, y2, x1, y2 - raio,
        x1, y1 + raio, x1, y1,
    ]


def iniciar_overlay():
    for widget in janela_config.winfo_children():
        widget.destroy()
    largura = janela_config.winfo_screenwidth()
    altura = janela_config.winfo_screenheight()
    janela_config.overrideredirect(True)
    janela_config.attributes("-topmost", True)
    janela_config.geometry(f"{largura}x{altura}+0+0")
    janela_config.configure(bg="black")
    try:
        janela_config.attributes("-transparentcolor", "black")
    except Exception:
        pass

    canvas = tk.Canvas(
        janela_config, width=largura, height=altura,
        bg="black", highlightthickness=0,
    )
    canvas.pack(fill="both", expand=True)

    fonte_original = tkfont.Font(root=janela_config, family="Segoe UI", size=10)
    fonte_traducao = tkfont.Font(
        root=janela_config, family="Segoe UI", size=12, weight="bold"
    )
    fonte_rotulo = tkfont.Font(
        root=janela_config, family="Segoe UI", size=8, weight="bold"
    )

    caixa_busca = canvas.create_rectangle(
        0, 0, 0, 0, outline="#f9e2af", width=2, state="hidden"
    )
    sombra = canvas.create_polygon(
        0, 0, 0, 0, fill="#000000", outline="", stipple="gray50",
        smooth=True, splinesteps=12, state="hidden",
    )
    cartao = canvas.create_polygon(
        0, 0, 0, 0, fill="#111827", outline="#64748b", width=1,
        stipple="gray50", smooth=True, splinesteps=12, state="hidden",
    )
    rotulo_original = canvas.create_text(
        0, 0, anchor="nw", text="ORIGINAL", fill="#94a3b8",
        font=fonte_rotulo, state="hidden",
    )
    texto_original = canvas.create_text(
        0, 0, anchor="nw", text="", fill="#e2e8f0",
        font=fonte_original, state="hidden",
    )
    separador = canvas.create_line(
        0, 0, 0, 0, fill="#475569", width=1, state="hidden"
    )
    rotulo_traducao = canvas.create_text(
        0, 0, anchor="nw", text="TRADUCAO", fill="#a6e3a1",
        font=fonte_rotulo, state="hidden",
    )
    texto_traducao = canvas.create_text(
        0, 0, anchor="nw", text="", fill="#ffffff",
        font=fonte_traducao, state="hidden",
    )

    ultimo_estado_cartao = None
    ultima_caixa_busca = None

    def atualizar_frame():
        nonlocal ultimo_estado_cartao, ultima_caixa_busca
        if not run.executando:
            return

        try:
            mx, my = run.pyautogui.position()
            metade = run.TAMANHO_CAIXA_MOUSE // 2
            tem_texto = bool(
                run.texto_original_global
                and run.traducao_atual_global
                and time.time() - run.ultimo_tempo_traducao
                <= run.TEMPO_TOLERANCIA_DETECCAO
            )

            if tem_texto:
                tx, ty = run.posicao_traducao_global
                original = run.texto_original_global
                traducao = run.traducao_atual_global
                largura_conteudo_maxima = 390
                original_quebrado = quebrar_linhas(
                    original, fonte_original, largura_conteudo_maxima
                )
                traducao_quebrada = quebrar_linhas(
                    traducao, fonte_traducao, largura_conteudo_maxima
                )
                linhas_original = original_quebrado.count("\n") + 1
                linhas_traducao = traducao_quebrada.count("\n") + 1
                largura_conteudo = min(
                    largura_conteudo_maxima,
                    max(
                        max(fonte_original.measure(linha) for linha in original_quebrado.split("\n")),
                        max(fonte_traducao.measure(linha) for linha in traducao_quebrado.split("\n")),
                        190,
                    ),
                )
                margem = 16
                largura_cartao = largura_conteudo + margem * 2
                altura_original = linhas_original * fonte_original.metrics("linespace")
                altura_traducao = linhas_traducao * fonte_traducao.metrics("linespace")
                topo_conteudo = 12
                y_rotulo_original = topo_conteudo
                y_texto_original = y_rotulo_original + fonte_rotulo.metrics("linespace") + 3
                y_separador = y_texto_original + altura_original + 8
                y_rotulo_traducao = y_separador + 9
                y_texto_traducao = y_rotulo_traducao + fonte_rotulo.metrics("linespace") + 3
                altura_cartao = y_texto_traducao + altura_traducao + 13

                # O cartao fica abaixo da area lida; se nao couber, vai acima.
                x1 = max(8, min(largura - largura_cartao - 8, tx - largura_cartao // 2))
                y1 = ty + metade + 12
                if y1 + altura_cartao > altura - 8:
                    y1 = ty - metade - altura_cartao - 12
                y1 = max(8, min(altura - altura_cartao - 8, y1))
                x2 = x1 + largura_cartao
                y2 = y1 + altura_cartao
                pontos = pontos_retangulo_arredondado(x1, y1, x2, y2)
                assinatura = (
                    original, traducao, x1, y1, largura_cartao, altura_cartao,
                    original_quebrado, traducao_quebrada,
                )

                canvas.itemconfigure(caixa_busca, state="normal")
                coords_busca = (tx - metade, ty - metade, tx + metade, ty + metade)
                if coords_busca != ultima_caixa_busca:
                    canvas.coords(caixa_busca, *coords_busca)
                    ultima_caixa_busca = coords_busca

                if assinatura != ultimo_estado_cartao:
                    canvas.coords(sombra, *pontos_retangulo_arredondado(
                        x1 + 4, y1 + 5, x2 + 4, y2 + 5
                    ))
                    canvas.coords(cartao, *pontos)
                    canvas.coords(rotulo_original, x1 + margem, y1 + y_rotulo_original)
                    canvas.coords(texto_original, x1 + margem, y1 + y_texto_original)
                    canvas.itemconfigure(texto_original, text=original_quebrado)
                    canvas.coords(separador, x1 + margem, y1 + y_separador,
                                  x2 - margem, y1 + y_separador)
                    canvas.coords(rotulo_traducao, x1 + margem, y1 + y_rotulo_traducao)
                    canvas.coords(texto_traducao, x1 + margem, y1 + y_texto_traducao)
                    canvas.itemconfigure(texto_traducao, text=traducao_quebrada)
                    for item in (
                        sombra, cartao, rotulo_original, texto_original,
                        separador, rotulo_traducao, texto_traducao,
                    ):
                        canvas.itemconfigure(item, state="normal")
                    ultimo_estado_cartao = assinatura
            else:
                canvas.itemconfigure(caixa_busca, state="normal")
                coords_busca = (mx - metade, my - metade, mx + metade, my + metade)
                if coords_busca != ultima_caixa_busca:
                    canvas.coords(caixa_busca, *coords_busca)
                    ultima_caixa_busca = coords_busca
                if ultimo_estado_cartao is not None:
                    for item in (
                        sombra, cartao, rotulo_original, texto_original,
                        separador, rotulo_traducao, texto_traducao,
                    ):
                        canvas.itemconfigure(item, state="hidden")
                    ultimo_estado_cartao = None
        except Exception as erro:
            print(f"[OVERLAY] {erro}")
        janela_config.after(30, atualizar_frame)

    atualizar_frame()


def encerrar():
    run.parar_tradutor()
    janela_config.destroy()


janela_config = tk.Tk()
janela_config.title("Tradutor - Modelos")
janela_config.geometry("760x520")
janela_config.resizable(False, False)
janela_config.protocol("WM_DELETE_WINDOW", encerrar)

abas = ttk.Notebook(janela_config)
aba_configuracao = ttk.Frame(abas, padding=10)
aba_huggingface = ttk.Frame(abas, padding=10)
abas.add(aba_configuracao, text="Configuracao local")
abas.add(aba_huggingface, text="Modelos Hugging Face")
abas.pack(fill="both", expand=True)

tk.Label(
    aba_configuracao, text="CONFIGURACAO DOS MODELOS",
    font=("Segoe UI", 14, "bold"),
).pack(pady=12)

itens = [
    ("det", "Deteccao OCR"),
    ("rec", "Reconhecimento OCR"),
    ("cls", "Classificacao OCR (opcional)"),
    ("keys", "Dicionario OCR (opcional)"),
    ("argos", "Traducao Argos"),
]

for tipo, nome in itens:
    linha = tk.Frame(aba_configuracao)
    linha.pack(fill="x", padx=15, pady=5)
    tk.Label(linha, text=nome, width=27, anchor="w").pack(side="left")
    campo = tk.Entry(linha)
    campo.pack(side="left", fill="x", expand=True, padx=5)
    campos[tipo] = campo
    tk.Button(
        linha, text="Abrir", command=lambda tipo=tipo: selecionar_arquivo(tipo)
    ).pack(side="right")

carregar_config()

tk.Label(
    aba_configuracao,
    text="Os caminhos sao salvos automaticamente em modelos.json",
    font=("Segoe UI", 9), fg="gray",
).pack(pady=10)

tk.Button(
    aba_configuracao, text="INICIAR TRADUTOR", command=iniciar_programa,
    font=("Segoe UI", 11, "bold"), height=2, bg="#a6e3a1",
).pack(fill="x", padx=20, pady=10)

tk.Label(
    aba_huggingface,
    text="BUSCAR MODELOS DISPONIVEIS NO HUGGING FACE",
    font=("Segoe UI", 14, "bold"),
).pack(pady=(4, 12))

linha_busca = tk.Frame(aba_huggingface)
linha_busca.pack(fill="x", pady=4)
tk.Label(linha_busca, text="Categoria:").pack(side="left")

categoria_huggingface_var = tk.StringVar(value="det")
combo_categoria = ttk.Combobox(
    linha_busca,
    textvariable=categoria_huggingface_var,
    values=list(CATEGORIAS_HUGGINGFACE),
    state="readonly",
    width=12,
)
combo_categoria.pack(side="left", padx=(6, 18))

tk.Label(linha_busca, text="Busca:").pack(side="left")
busca_huggingface = tk.Entry(linha_busca)
busca_huggingface.insert(0, CATEGORIAS_HUGGINGFACE["det"]["busca"])
busca_huggingface.pack(side="left", fill="x", expand=True, padx=6)


def trocar_categoria_huggingface(_evento=None):
    global categoria_huggingface
    categoria_huggingface = categoria_huggingface_var.get()
    busca_huggingface.delete(0, tk.END)
    busca_huggingface.insert(0, CATEGORIAS_HUGGINGFACE[categoria_huggingface]["busca"])


combo_categoria.bind("<<ComboboxSelected>>", trocar_categoria_huggingface)
tk.Button(
    linha_busca, text="Buscar", command=iniciar_busca_huggingface,
).pack(side="right")

lista_huggingface = tk.Listbox(aba_huggingface, height=14)
lista_huggingface.pack(fill="both", expand=True, pady=8)
lista_huggingface.bind("<<ListboxSelect>>", selecionar_resultado_huggingface)

arquivo_selecionado_huggingface = tk.StringVar()
tk.Label(aba_huggingface, text="Arquivo selecionado:").pack(anchor="w")
tk.Entry(
    aba_huggingface, textvariable=arquivo_selecionado_huggingface,
    state="readonly",
).pack(fill="x", pady=(2, 8))

tk.Button(
    aba_huggingface, text="BAIXAR E USAR ESTE MODELO",
    command=baixar_modelo_huggingface,
    font=("Segoe UI", 10, "bold"), bg="#a6e3a1",
).pack(fill="x", pady=4)

status_huggingface = tk.StringVar(value="Escolha uma categoria e clique em Buscar.")
tk.Label(
    aba_huggingface, textvariable=status_huggingface,
    anchor="w", fg="gray",
).pack(fill="x", pady=(6, 0))


if __name__ == "__main__":
    janela_config.mainloop()
