import os
if not df_view.empty:
for i, row in df_view.iterrows():
col1, col2, col3, col4 = st.columns([4,1,1,1])
with col1:
st.write(f"**{row['canonical_filename']}**")
st.caption(row['drive_path'])
with col2:
# Download individual
try:
file_abs = os.path.join(row['drive_path'])
# se path é relativo, tornar absoluto
if not os.path.isabs(file_abs):
file_abs = os.path.join(STORAGE_ROOT, os.path.relpath(file_abs, start=STORAGE_ROOT))
if os.path.exists(file_abs):
with open(file_abs, 'rb') as f:
st.download_button("Baixar", data=f.read(), file_name=os.path.basename(file_abs), key=f"dl_{i}")
else:
st.button("Baixar", disabled=True, key=f"dl_{i}")
except Exception:
st.button("Baixar", disabled=True, key=f"dl_{i}")
with col3:
# Excluir
if st.button("Excluir", key=f"rm_{i}"):
try:
file_abs = os.path.join(row['drive_path'])
if not os.path.isabs(file_abs):
file_abs = os.path.join(STORAGE_ROOT, os.path.relpath(file_abs, start=STORAGE_ROOT))
if os.path.exists(file_abs):
os.remove(file_abs)
# limpar diretórios vazios acima (até @Atletas/<slug>) — best-effort
try:
parent = os.path.dirname(file_abs)
for _ in range(3):
if parent.startswith(STORAGE_ROOT) and os.path.isdir(parent) and not os.listdir(parent):
os.rmdir(parent)
parent = os.path.dirname(parent)
except Exception:
pass
# remover do index/regatta e master
def pred(s):
return s['canonical_filename'] == row['canonical_filename'] and s['file_hash_sha256'] == row['file_hash_sha256']
remove_row_csv(index_csv_path, pred)
master_csv_path = os.path.join(STORAGE_ROOT, MASTER_CSV_NAME)
remove_row_csv(master_csv_path, pred)
st.success("Arquivo excluído e inventário atualizado.")
except Exception as e:
st.error(f"Falha ao excluir: {e}")
with col4:
# Mostrar hash curto
st.code(row['file_hash_sha256'][:8], language=None)
else:
st.info("Nenhum arquivo para este filtro.")