"""Display reported facts without claiming unperformed geometry checks."""
def healing_result(summary):
    numbers = summary.get("numbers", {})
    return {
        "artifact": "run/healed.stp", "source": "run/summary.json",
        "execution_status": summary.get("stages", {}).get("heal", {}).get("status", "unknown"),
        "closed": numbers.get("closed"), "valid": numbers.get("valid"),
        "floating_caps": numbers.get("floating_caps"),
        "holes_found": numbers.get("holes_found"), "holes_filled": numbers.get("holes_filled"),
        "holes_left": numbers.get("holes_left"),
        "free_boundaries_measured": numbers.get("free_boundaries_measured"),
        "note": "힐링 단계 보고값입니다. 실행 성공은 형상 품질 통과를 뜻하지 않습니다. STEP 되읽기·경계상자 재검사는 이 표시에서 수행하지 않았습니다. 최종 랩·캡 결과와 별개입니다."
    }
