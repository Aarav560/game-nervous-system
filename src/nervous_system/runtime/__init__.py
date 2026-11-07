"""Runtime: message bus + main loop (Architecture §10, §11).

Main loop sketch (per bot):
    every tick:      reflexes.tick -> adapter.act; autonomic rules; forward model; salience
    every cadence:   adapter.describe -> laya.decide -> EscalationPolicy -> reflex or escalate
    on escalation / salient event / goal expiry / heartbeat: brain
    always:          watchdogs; kill switch checked first
"""
