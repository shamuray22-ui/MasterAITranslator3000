import json
import threading
import time
import tkinter as tk
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

    def atualizar_frame():
        canvas.delete("all")
        if not run.executando:
            return
        try:
            mx, my = run.pyautogui.position()
            metade = run.TAMANHO_CAIXA_MOUSE // 2
            if time.time() - run.ultimo_tempo_traducao > 2.5:
                run.traducao_atual_global = ""
            if run.traducao_atual_global:
                tx, ty = run.posicao_traducao_global
                canvas.create_rectangle(
                    tx - metade, ty - metade, tx + metade, ty + metade,
                    fill="black", outline="#a6e3a1", width=2,
                )
                canvas.create_text(
                    tx, ty, text=run.traducao_atual_global,
                    fill="#ffffff", anchor="center", width=180,
                    font=("Segoe UI", 9, "bold"),
                )
            else:
                canvas.create_rectangle(
                    mx - metade, my - metade, mx + metade, my + metade,
                    outline="#f9e2af", width=2,
                )
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
