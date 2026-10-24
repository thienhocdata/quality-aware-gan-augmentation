"""Build the current CPU-only master notebook from the preserved 17/10 master.

The added sections only read the saved endpoint/V3-A bundles and the source
registry. They never import TensorFlow, train, restore a model, or rerun test
inference.
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "notebooks" / "00_MASTER.ipynb"
TARGET = ROOT / "notebooks" / "00_MASTER_CURRENT.ipynb"


TITLE = """# KLTN — 3D-HyperGAMO và Learned-local V2 trên Indian Pines
## Notebook master hiện hành: source đã khóa, output đã lưu và các ablation cuối

**Protocol B `50e988fae260` · 16 lớp · patch 11×11 · PCA 40 · seed 42/43/44.**

Notebook này nối các phần theo trục nghiên cứu hiện hành. Source gốc được giữ theo hash; kết quả lấy từ output đã chạy. **Run All mặc định không train, không chạy lại test và không khởi tạo TensorFlow.**

| Phần | Trạng thái |
|---|---|
| 1. Protocol B và baseline lịch sử | Đọc output đã lưu |
| 2. Kiến trúc, graph và lịch M4 | Source thực thi đã khóa |
| 3. M4 bốn nhánh × ba seed | Hoàn tất 12/12 tại 8.000 C updates |
| 4. Test M4 gốc | Hoàn tất 12/12; chỉ đọc predictions đã lưu |
| 5. Diagnostics phân loại | Phân tích hậu nghiệm |
| 6. Learned weighting 8.000 | Hoàn tất ba seed |
| 7. Endgame 7.500→8.000 | Có checkpoint, chưa có report forward COMPLETE |
| 8. Endpoint ablation | Hoàn tất sáu classifier |
| 9. V3-A distance-bias | Hoàn tất ba seed; ablation âm |
| 10. Provenance/source | 22/22 source và 12/12 artifact PASS |
| 11. Kết luận | Giới hạn và phạm vi báo cáo |

Test Indian Pines đã được xem trong quá trình phát triển; endpoint/V3-A là phân tích hậu nghiệm. Nhánh leakage được giữ như giới hạn protocol, không trộn vào bảng M4 như một cùng thí nghiệm.
"""


ENDPOINT_MD = """## Phần 8 — Controlled endpoint ablation

Sáu classifier C-only so sánh `learned_weights` với `random_one_fixed` ở ba cặp seed. Phần này đọc predictions/metrics đã lưu, không restore G/C và không inference lại.

Đây trả lời câu hỏi trong phạm vi endpoint-training ablation: giữ endpoint G/head và thay quy tắc chọn nguồn trong quá trình train C. Nó không chứng minh quan hệ nhân quả cho toàn bộ quá trình GAN.
"""


ENDPOINT_CODE = r'''import io, zipfile

def find_saved_bundle(filename):
    candidates = []
    if LOCAL_PROJECT is not None:
        direct = LOCAL_PROJECT / 'outputs' / 'bundles' / filename
        if direct.is_file(): candidates.append(direct)
    for root in INPUT_ROOTS:
        if root.is_file() and root.name == filename:
            candidates.append(root)
        elif root.is_dir():
            candidates.extend(root.rglob(filename))
    unique = {str(p.resolve()): p for p in candidates}
    if len(unique) != 1:
        raise RuntimeError(f'Không định danh duy nhất {filename}: {list(unique)}')
    return next(iter(unique.values()))

def read_csv_member(bundle, member):
    with zipfile.ZipFile(bundle) as archive:
        return pd.read_csv(io.BytesIO(archive.read(member)))

endpoint_bundle = find_saved_bundle('07_SELECTOR_ENDPOINT_ABLATION_ANALYSIS_20261019.zip')
endpoint_dir = OUTPUT_ROOT / '08_SELECTOR_ENDPOINT_ABLATION'
endpoint_dir.mkdir(parents=True, exist_ok=True)
endpoint_seed = read_csv_member(endpoint_bundle, 'metrics_per_seed_with_GM.csv')
endpoint_summary = read_csv_member(endpoint_bundle, 'summary_mean_sample_SD.csv')
endpoint_paired = read_csv_member(endpoint_bundle, 'paired_differences_recomputed.csv')
endpoint_seed.to_csv(endpoint_dir/'metrics_per_seed.csv', index=False)
endpoint_summary.to_csv(endpoint_dir/'summary_mean_sample_SD.csv', index=False)
endpoint_paired.to_csv(endpoint_dir/'paired_differences.csv', index=False)
display(Markdown('**Kết quả từng cặp seed**'))
display(endpoint_seed)
display(Markdown('**Trung bình ± sample SD và chênh ghép cặp**'))
display(endpoint_summary)
display(endpoint_paired)
ENDPOINT_STATUS = {
    'status': 'COMPLETE_6_CLASSIFIERS_SAVED_RESULTS',
    'bundle': str(endpoint_bundle),
    'new_training_performed': False,
    'new_test_inference_performed': False,
}
'''


V3A_MD = """## Phần 9 — V3-A distance-bias: ablation âm

V3-A thêm distance bias trainable vào logits selector như một thử nghiệm khắc phục có kiểm soát. Ba seed đã chạy đủ. Phần này đọc kết quả đã lưu và giữ V2 là phương pháp chính vì V3-A không cải thiện Macro-F1.
"""


V3A_CODE = r'''v3a_bundle = find_saved_bundle('08_V3A_RESULTS_ANALYSIS_20261019.zip')
v3a_dir = OUTPUT_ROOT / '09_V3A_NEGATIVE_ABLATION'
v3a_dir.mkdir(parents=True, exist_ok=True)
v3a_summary = read_csv_member(v3a_bundle, 'recomputed_summary_mean_sample_SD.csv')
v3a_delta = read_csv_member(v3a_bundle, 'V3_minus_full_M4_V2_by_seed.csv')
v3a_probe = read_csv_member(v3a_bundle, 'selector_probe_summary.csv')
v3a_margin = read_csv_member(v3a_bundle, 'distance_margin_evidence.csv')
for frame, name in [(v3a_summary,'summary_mean_sample_SD.csv'),(v3a_delta,'V3_minus_V2_by_seed.csv'),(v3a_probe,'selector_probe_summary.csv'),(v3a_margin,'distance_margin_evidence.csv')]:
    frame.to_csv(v3a_dir/name,index=False)
display(Markdown('**So sánh 5 phương pháp từ saved predictions**'))
display(v3a_summary)
display(Markdown('**V3-A trừ V2 theo seed**'))
display(v3a_delta)
display(Markdown('**Hành vi selector V3-A tại 200 và 8.000 updates**'))
display(v3a_probe)
V3A_STATUS = {
    'status': 'COMPLETE_3_SEEDS_NEGATIVE_ABLATION',
    'bundle': str(v3a_bundle),
    'new_training_performed': False,
    'new_test_inference_performed': False,
}
'''


PROVENANCE_MD = """## Phần 10 — Source, artifact và cấu trúc project

Registry hiện hành phân biệt source đã thực thi, source chẩn đoán, notebook tham chiếu và output. Cell này chỉ đọc manifest; không nạp mô hình.
"""


PROVENANCE_CODE = r'''CURRENT_REGISTRY_STATUS = {'status':'UNAVAILABLE_ON_THIS_HOST'}
manifest_candidates = []
if LOCAL_PROJECT is not None:
    manifest_candidates.append(LOCAL_PROJECT/'CURRENT_PROJECT_MANIFEST.json')
for root in INPUT_ROOTS:
    if root.is_dir(): manifest_candidates.extend(root.rglob('CURRENT_PROJECT_MANIFEST.json'))
manifest_candidates = [p for p in manifest_candidates if p.is_file()]
if manifest_candidates:
    current_manifest_path = manifest_candidates[0]
    CURRENT_REGISTRY_STATUS = json.loads(current_manifest_path.read_text(encoding='utf-8'))
    display(pd.DataFrame([{
        'source_records': CURRENT_REGISTRY_STATUS.get('source_records'),
        'artifact_records': CURRENT_REGISTRY_STATUS.get('artifact_records'),
        'source_pass': CURRENT_REGISTRY_STATUS.get('source_identity_checks_pass'),
        'artifact_pass': CURRENT_REGISTRY_STATUS.get('artifact_presence_checks_pass'),
        'protocol': CURRENT_REGISTRY_STATUS.get('protocol_B_id'),
    }]))
    display(FileLink(str(current_manifest_path)))
else:
    print('SKIP registry: gói Kaggle này không kèm manifest project; kết quả khoa học đã lưu vẫn có thể đọc.')
'''


CONCLUSION = """## Phần 11 — Kết luận và giới hạn

- Learned-local V2 là phương pháp chính hiện tại: Macro-F1 trung bình cao hơn Original GAN, Matched ROS và Random-local trong M4, nhưng chênh lệch theo seed/lớp chưa ổn định tuyệt đối.
- Endpoint ablation cho learned selector thắng random-one-fixed ở 2/3 cặp seed và tăng trung bình Macro-F1, nhưng không thay thế so sánh toàn hệ thống M4.
- V3-A đã chạy đủ ba seed nhưng thấp hơn V2; được giữ như ablation âm, không thay V2.
- Protocol B vẫn có spatial/raw-source overlap. Các phép thử leakage trước đây chưa cô lập được một mức ảnh hưởng nhân quả chắc chắn lên accuracy.
- Test Indian Pines đã được xem. Mọi phân tích/biến thể sau đó phải ghi là post-hoc motivated; không tuyên bố SOTA hay generalization từ một cảnh.
- Endgame 7.500→8.000 có source/checkpoint nhưng chưa có report COMPLETE; không tự suy diễn kết quả.

Cell cuối lưu manifest của lần review CPU này. Các output gốc và checkpoint không bị thay đổi.
"""


def main() -> None:
    notebook = nbf.read(SOURCE, as_version=4)
    notebook.cells[0].source = TITLE
    conclusion_index = next(i for i, cell in enumerate(notebook.cells) if cell.cell_type == "markdown" and "Kết luận và giới hạn" in cell.source)
    additions = [
        nbf.v4.new_markdown_cell(ENDPOINT_MD),
        nbf.v4.new_code_cell(ENDPOINT_CODE),
        nbf.v4.new_markdown_cell(V3A_MD),
        nbf.v4.new_code_cell(V3A_CODE),
        nbf.v4.new_markdown_cell(PROVENANCE_MD),
        nbf.v4.new_code_cell(PROVENANCE_CODE),
    ]
    notebook.cells[conclusion_index:conclusion_index] = additions
    for cell in notebook.cells:
        if cell.cell_type == "markdown" and cell.source.startswith("## Phần 8 — Kết luận"):
            cell.source = CONCLUSION
        if cell.cell_type == "code" and "final_status = {" in cell.source:
            cell.source = cell.source.replace(
                "    'endgame_session':ENDGAME_RESULT,\n",
                "    'endgame_session':ENDGAME_RESULT,\n"
                "    'endpoint_ablation':ENDPOINT_STATUS,\n"
                "    'v3a_negative_ablation':V3A_STATUS,\n"
                "    'current_registry':CURRENT_REGISTRY_STATUS,\n",
            )
    notebook.metadata["kltn_master_revision"] = "2026-10-23-current"
    notebook.metadata["default_execution"] = "CPU saved-results review; no TensorFlow/model execution"
    nbf.write(notebook, TARGET)
    print(TARGET.name)


if __name__ == "__main__":
    main()
