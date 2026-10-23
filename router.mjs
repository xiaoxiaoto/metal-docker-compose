import querystring from 'querystring';

const targets = {
    'h2o-lightning-4b': {location: '@h2o', limit: 25 * 1024 * 1024},
    'higgs-audio-v3-tts-4b': {location: '@higgs', limit: 16 * 1024},
    'yue2-3b': {location: '@yue2', limit: 128 * 1024},
    'moss-transcribe-diarize': {location: '@moss', limit: 25 * 1024 * 1024},
    'paddleocr-vl': {location: '@paddleocr', limit: 25 * 1024 * 1024},
    'qwen-image-2.1-turbo': {location: '@qwenimage', limit: 64 * 1024},
    'seedvr2': {location: '@seedvr2', limit: 25 * 1024 * 1024},
};

function multipartModel(body, contentType) {
    const match = /;\s*boundary\s*=\s*(?:"([^"]+)"|([^;\s]+))/i.exec(contentType);
    const boundary = match && (match[1] || match[2]);
    if (!boundary || !/^[0-9A-Za-z'()+_,./:=? -]{1,70}$/.test(boundary) || boundary.endsWith(' ')) {
        throw new Error('Invalid multipart boundary');
    }
    const delimiter = Buffer.from('--' + boundary);
    const separator = Buffer.from('\r\n--' + boundary);
    let position = body.indexOf(delimiter);
    let model;
    if (position < 0 || (position > 0 && body.subarray(position - 2, position).toString() !== '\r\n')) {
        throw new Error('Invalid multipart body');
    }
    while (true) {
        position += delimiter.length;
        const suffix = body.subarray(position, position + 2).toString();
        if (suffix === '--') {
            return model;
        }
        if (suffix !== '\r\n') {
            throw new Error('Invalid multipart delimiter');
        }
        const start = position + 2;
        const headerEnd = body.indexOf('\r\n\r\n', start);
        if (headerEnd < 0 || headerEnd - start > 8192) {
            throw new Error('Invalid multipart headers');
        }
        const valueStart = headerEnd + 4;
        let end = body.indexOf(separator, valueStart);
        while (end >= 0) {
            const after = end + separator.length;
            const tail = body.subarray(after, after + 2).toString();
            if (tail === '\r\n' || (tail === '--' &&
                (after + 2 === body.length || body.subarray(after + 2, after + 4).toString() === '\r\n'))) {
                break;
            }
            end = body.indexOf(separator, end + 1);
        }
        if (end < 0) {
            throw new Error('Incomplete multipart body');
        }
        const headers = body.subarray(start, headerEnd).toString();
        const disposition = /^content-disposition:\s*form-data\s*;([^\r\n]*)/im.exec(headers);
        const name = disposition && /(?:^|;)\s*name\s*=\s*(?:"([^"]*)"|([^;\s]+))/i.exec(disposition[1]);
        if (name && (name[1] || name[2]) === 'model') {
            if (model !== undefined || /(?:^|;)\s*filename(?:\*)?\s*=/i.test(disposition[1])) {
                throw new Error('model must be a single text field');
            }
            if (end - valueStart > 256) {
                throw new Error('model is too long');
            }
            model = body.subarray(valueStart, end).toString('utf8');
        }
        position = end + 2;
    }
}

function error(r, status, message) {
    r.headersOut['Content-Type'] = 'application/json';
    r.return(status, JSON.stringify({error: {message, type: 'invalid_request_error', param: 'model', code: status}}));
}

function route(r) {
    const body = r.requestBuffer;
    const contentType = r.headersIn['Content-Type'] || '';
    const mediaType = contentType.split(';')[0].trim().toLowerCase();
    let model;
    try {
        if (body && body.length) {
            if (mediaType === 'application/json') {
                const data = JSON.parse(body.toString('utf8'));
                if (!data || typeof data !== 'object' || Array.isArray(data)) {
                    throw new Error('JSON body must be an object');
                }
                model = data.model;
            } else if (mediaType === 'multipart/form-data') {
                model = multipartModel(body, contentType);
            } else if (mediaType === 'application/x-www-form-urlencoded') {
                model = querystring.parse(body.toString('utf8')).model;
            } else {
                return error(r, 415, 'Use JSON, multipart/form-data or application/x-www-form-urlencoded');
            }
        }
        const queryModel = r.args.model;
        if (queryModel !== undefined) {
            if (Array.isArray(queryModel) || (model !== undefined && queryModel !== model)) {
                throw new Error('Ambiguous model parameter');
            }
            model = queryModel;
        }
        if (typeof model !== 'string' || !model.length || model.length > 256) {
            throw new Error('A single non-empty model parameter is required');
        }
    } catch (exception) {
        return error(r, 400, exception.message);
    }
    const target = Object.prototype.hasOwnProperty.call(targets, model) ? targets[model] : undefined;
    if (!target) {
        return error(r, 404, 'Unknown model: ' + model);
    }
    if (body && body.length > target.limit) {
        return error(r, 413, 'Request body exceeds the limit for this model');
    }
    if (r.uri === '/v1/decisions/') {
        return error(r, 404, 'Use the exact /v1/decisions path without a trailing slash');
    }
    if (r.uri === '/v1/decisions' && model !== 'h2o-lightning-4b') {
        return error(r, 400, '/v1/decisions is not supported for model: ' + model);
    }
    // Named locations preserve the original method, URI, query and binary request body.
    r.internalRedirect(target.location);
}

async function models(r) {
    if (r.args.model !== undefined) {
        return route(r);
    }
    if (r.method !== 'GET') {
        r.headersOut.Allow = 'GET';
        return error(r, 405, 'Use GET /v1/models');
    }
    const results = await Promise.all(Object.keys(targets).map(async (id) => {
        try {
            const reply = await r.subrequest('/_models/' + targets[id].location.slice(1), {method: 'GET', args: '', body: ''});
            if (reply.status !== 200) {
                throw new Error('HTTP ' + reply.status);
            }
            const list = JSON.parse(reply.responseText);
            if (list.object !== 'list' || !Array.isArray(list.data)) {
                throw new Error('Invalid model list');
            }
            const model = list.data.find((item) => item && item.id === id);
            if (!model || model.object !== 'model' || !Number.isInteger(model.created) || model.created < 0 ||
                typeof model.owned_by !== 'string' || !model.owned_by.length) {
                throw new Error('Missing or invalid model metadata');
            }
            return {id: model.id, object: 'model', created: model.created, owned_by: model.owned_by};
        } catch (exception) {
            r.warn('Cannot list model ' + id + ': ' + exception.message);
            return null;
        }
    }));
    r.headersOut['Content-Type'] = 'application/json';
    r.return(200, JSON.stringify({object: 'list', data: results.filter((model) => model !== null)}));
}

export default {route, models};
