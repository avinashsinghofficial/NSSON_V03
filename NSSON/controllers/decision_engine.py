def choose_target(local_score, spike_score, roi_bytes, local_load, telemetry, nodes):
    jetson = nodes.get('jetson-nano-1', {'status':'down','load':1.0,'url':''})
    remote = nodes.get('remote-verifier-1', {'status':'down','load':1.0,'url':''})
    uncertainty = abs(local_score - 0.5)
    urgency = max(local_score, spike_score)
    if urgency > 0.80 and remote.get('status') == 'up':
        target = 'remote'
    elif jetson.get('status') == 'up' and jetson.get('load',1.0) < 0.85 and telemetry.get('link_delay_ms',999) < 40:
        target = 'jetson' if (uncertainty < 0.22 or local_load > 0.75 or roi_bytes < 120000) else 'local'
    else:
        target = 'local'
    qos = 'critical' if urgency > 0.80 else 'priority' if urgency > 0.60 else 'best_effort'
    return {'target': target, 'qos_class': qos, 'route_id': f'{target}-{qos}', 'telemetry': telemetry, 'jetson_url': jetson.get('url',''), 'remote_url': remote.get('url','')}
