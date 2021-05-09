def do_POST(s):
    response_code = 200
    response = ""
    var_len = int(s.headers.get('Content-Length'))
    content = s.rfile.read(var_len)
    payload = json.loads(content)

    if payload.get('train'):
        nn.train(payload['trainArray'])
        nn.save
    elif payload.get('predict'):
        try:
            response = {
                'type': 'test',
                'result': nn.predict(str(payload['image']))
            }
        except:
            response_code = 500
    else:
        response_code = 400

    s.send_response(response_code)
    s.send_header("Content-type", "application/json")
    s.send_header("Access-Control-Allow-Origin", "*")
    s.end_headers()
    
    if response:
        s.wfile.write(json.dumps(response))
    return