from __future__ import annotations

import json

from freeman.interface.cli import main


def test_cli_run_and_query(schema_path, lite_config_path, capsys) -> None:
    exit_code = main(["run", "--schema", str(schema_path), "--config", str(lite_config_path)])
    assert exit_code == 0
    run_payload = json.loads(capsys.readouterr().out)
    assert run_payload["status"] == "compiled"

    exit_code = main(["query", "water crisis", "--config", str(lite_config_path)])
    assert exit_code == 0
    query_payload = json.loads(capsys.readouterr().out)
    assert query_payload["matched"] is True
