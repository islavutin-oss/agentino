"""Tests for agentino.gates — workflow gate enforcement."""

from agentino.safety.gates import GateManager, GateRule


def test_no_rules_allows_all():
    gm = GateManager()
    assert gm.check("any_tool") is None


def test_unmarked_gate_rejects():
    rules = [GateRule(gate="security_checked", tools=["send_email"], message="Scan first.")]
    gm = GateManager(rules)
    assert gm.check("send_email") == "Scan first."


def test_marked_gate_allows():
    rules = [GateRule(gate="security_checked", tools=["send_email"], message="Scan first.")]
    gm = GateManager(rules)
    gm.mark("security_checked")
    assert gm.check("send_email") is None


def test_unrelated_tool_not_blocked():
    rules = [GateRule(gate="security_checked", tools=["send_email"], message="Scan first.")]
    gm = GateManager(rules)
    assert gm.check("read_file") is None


def test_conditional_gate_skips_when_condition_not_met():
    rules = [
        GateRule(
            gate="security_checked",
            tools=["send_email"],
            message="Scan first.",
            condition="external_content_read",
        )
    ]
    gm = GateManager(rules)
    # condition not met → rule not enforced → allowed
    assert gm.check("send_email") is None


def test_conditional_gate_enforces_when_condition_met():
    rules = [
        GateRule(
            gate="security_checked",
            tools=["send_email"],
            message="Scan first.",
            condition="external_content_read",
        )
    ]
    gm = GateManager(rules)
    gm.mark("external_content_read")
    # condition met, gate not marked → rejected
    assert gm.check("send_email") == "Scan first."


def test_conditional_gate_passes_when_both_met():
    rules = [
        GateRule(
            gate="security_checked",
            tools=["send_email"],
            message="Scan first.",
            condition="external_content_read",
        )
    ]
    gm = GateManager(rules)
    gm.mark("external_content_read")
    gm.mark("security_checked")
    assert gm.check("send_email") is None


def test_multiple_tools_per_rule():
    rules = [GateRule(gate="auth", tools=["send_email", "write_record"], message="Auth first.")]
    gm = GateManager(rules)
    assert gm.check("send_email") == "Auth first."
    assert gm.check("write_record") == "Auth first."
    gm.mark("auth")
    assert gm.check("send_email") is None
    assert gm.check("write_record") is None


def test_multiple_rules_all_must_pass():
    rules = [
        GateRule(gate="security_checked", tools=["send_email"], message="Scan first."),
        GateRule(gate="auth_checked", tools=["send_email"], message="Auth first."),
    ]
    gm = GateManager(rules)
    gm.mark("security_checked")
    assert gm.check("send_email") == "Auth first."
    gm.mark("auth_checked")
    assert gm.check("send_email") is None


def test_reset_clears_all():
    rules = [GateRule(gate="security_checked", tools=["send_email"], message="Scan first.")]
    gm = GateManager(rules)
    gm.mark("security_checked")
    gm.track("email_id", "123")
    gm.reset()
    assert gm.check("send_email") == "Scan first."
    assert gm.get_tracked("email_id") == ""


def test_track_and_retrieve():
    gm = GateManager()
    gm.track("current_email", "msg_42")
    assert gm.get_tracked("current_email") == "msg_42"
    assert gm.get_tracked("missing", "default") == "default"


def test_is_marked():
    gm = GateManager()
    assert not gm.is_marked("security_checked")
    gm.mark("security_checked")
    assert gm.is_marked("security_checked")


def test_marked_gates_returns_copy():
    gm = GateManager()
    gm.mark("a")
    gm.mark("b")
    gates = gm.marked_gates
    assert gates == {"a", "b"}
    gates.add("c")  # mutating copy shouldn't affect original
    assert gm.marked_gates == {"a", "b"}
