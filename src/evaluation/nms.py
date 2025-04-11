"""Implement Non-Maximum Suppression (NMS) to remove duplicate predictions that are too close together in time."""

def nms(preds, nms_thresh):
    preds_after_nms = {}
    for m, p in preds.items():
        if m not in preds_after_nms:
            preds_after_nms[m] = {"UrlLocal": m, "predictions": []}

        cur_label = "start"
        cur_time = "0"
        cur_score = "0"
        cur_gameTime = ""
        cur_half = ""

        for i, pred in enumerate(p["predictions"]):
            if pred["label"] == cur_label and (int(pred["position"]) - int(cur_time)) < nms_thresh and float(pred["confidence"]) > float(cur_score) and i > 0:
                preds_after_nms[m]["predictions"].remove({"gameTime": cur_gameTime, "label": cur_label, "position": cur_time, "confidence": cur_score, "half": cur_half})
                cur_label = pred["label"]
                cur_time = pred["position"]
                cur_score = pred["confidence"]
                cur_gameTime = pred["gameTime"]
                cur_half = pred["half"]
                preds_after_nms[m]["predictions"].append(pred)
            elif pred["label"] == cur_label and (int(pred["position"]) - int(cur_time)) < nms_thresh and float(pred["confidence"]) <= float(cur_score) and i > 0:
                continue
            else:
                cur_label = pred["label"]
                cur_time = pred["position"]
                cur_score = pred["confidence"]
                cur_gameTime = pred["gameTime"]
                cur_half = pred["half"]
                preds_after_nms[m]["predictions"].append(pred)

    return preds_after_nms

def standard_nms(preds, nms_thresh):
    preds_after_nms = {}
    for m, p in preds.items():
        if m not in preds_after_nms:
            preds_after_nms[m] = {"UrlLocal": m, "predictions": []}

        p["predictions"].sort(key=sort_function, reverse=True)

        for i, pred in enumerate(p["predictions"]):
            if i == 0:
                preds_after_nms[m]["predictions"].append(pred)
            else:
                for stored in preds_after_nms[m]["predictions"]: #all the stored events have an higher score than the current pred (see the sort function!)
                    if pred["half"]==stored["half"] and pred["label"]==stored["label"] and abs(int(pred["position"])-int(stored["position"]))<nms_thresh:
                        break
                else:
                    preds_after_nms[m]["predictions"].append(pred)

    return preds_after_nms

def sort_function(event):
    return (event["label"], event["confidence"])

