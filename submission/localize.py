#!/usr/bin/env python3
 
import argparse
import time
 
import cv2
import numpy as np
GRAY_WEIGHT = 0.7
EDGE_WEIGHT = 0.3
def rotate_image(image, angle):
 
    h, w = image.shape
 
    center = (w // 2, h // 2)
 
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
 
    rotated = cv2.warpAffine(
        image,
        matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
 
    return rotated
 
SCALES = [
    9.0,
    9.25,
    9.5,
    9.75,
    10.0,
    10.25,
    10.5,
    10.75,
    11.0,
]
ANGLES = [-2, -1, 0, 1, 2]
 
def predict(reference_path, search_path):
 
    reference = cv2.imread(reference_path, cv2.IMREAD_GRAYSCALE)
    search = cv2.imread(search_path, cv2.IMREAD_GRAYSCALE)
    reference_edges = cv2.Canny(reference, 50, 150)
    search_edges = cv2.Canny(search, 50, 150)
    reference_edges = cv2.Canny(reference, 50, 150)
    search_edges = cv2.Canny(search, 50, 150)
    if reference is None:
        raise ValueError("Cannot read reference image")
 
    if search is None:
        raise ValueError("Cannot read search image")
 
    start = time.time()
 
    candidates = []
 
    for scale in SCALES:
 
        tw = max(1, int(reference.shape[1] / scale))
        th = max(1, int(reference.shape[0] / scale))
 
        if tw >= search.shape[1]:
            continue
 
        if th >= search.shape[0]:
            continue
 
        small_template = cv2.resize(

            reference,

            (tw, th),

            interpolation=cv2.INTER_AREA,   )
 
        
        template = small_template
        template_edges = cv2.Canny(template, 50, 150)

        gray_result = cv2.matchTemplate(
            search,
            template,
            cv2.TM_CCOEFF_NORMED,
        )
        
        edge_result = cv2.matchTemplate(
            search_edges,
            template_edges,
            cv2.TM_CCOEFF_NORMED,
        )
        
        combined = GRAY_WEIGHT* gray_result + EDGE_WEIGHT * edge_result
        
        _, score, _, max_loc = cv2.minMaxLoc(combined)
            
        candidates.append({

                "score": float(score),

                "scale": scale,

                "x": max_loc[0] + tw/2,

                "y": max_loc[1] + th/2,

                "left": max_loc[0],

                "top": max_loc[1],

                "width": tw,

                "height": th

        })
 
    
        
 
    candidates.sort(
        key=lambda x: x["score"],
        reverse=True,
    )
 
    top5 = candidates[:5]
    if len(top5) >= 2:
        gap = top5[0]["score"] - top5[1]["score"]
        print(f"Confidence Gap = {gap:.4f}")
    best = None
    best_score = top5[0]["score"]
    
    for c in top5:
        if best_score - c["score"] > 0.05:
            continue
        margin = 40
    
        left = max(c["left"] - margin, 0)

        top = max(c["top"] - margin, 0)
    
        right = min(

            c["left"] + c["width"] + margin,

            search.shape[1],

        )
    
        bottom = min(

            c["top"] + c["height"] + margin,

            search.shape[0],

        )
    
        crop = search[top:bottom, left:right]
        for scale_offset in [0.98,1.00,1.02]:
            w = int(c["width"] * scale_offset)
            h = int(c["height"] * scale_offset)
        

            
            template = cv2.resize(

                reference,

                (c["width"], c["height"]),

                interpolation=cv2.INTER_AREA,

            )
        template_edge = cv2.Canny(template, 50, 150)
        crop_edges = search_edges[top: bottom, left: right]

        
        
        
        for angle in ANGLES:
    
            rot = rotate_image(template, angle)
            rot_edge = rotate_image(template_edge, angle)
            gray_result = cv2.matchTemplate(
                crop,
                rot,
                cv2.TM_CCOEFF_NORMED,
            )
            
            edge_result = cv2.matchTemplate(
                crop_edges,
                rot_edge,
                cv2.TM_CCOEFF_NORMED,
            )
            
            combined_result = 0.7 * gray_result + 0.3 * edge_result
            
            _, score, _, max_loc = cv2.minMaxLoc(combined_result)

        
    
    
            real_left = left + max_loc[0]

            real_top = top + max_loc[1]
    
            if best is None or score > best["score"]:
    
                best = {

                    "score": float(score),

                    "scale": c["scale"],

                    "angle": angle,

                    "left": real_left,

                    "top": real_top,

                    "width": c["width"],

                    "height": c["height"],

                    "x": real_left + c["width"]/2,

                    "y": real_top + c["height"]/2,

                }
 
    runtime = time.time() - start

 
    print("\n==============================")
    print("TOP 5 MATCHES")
    print("==============================")
 
    for i, c in enumerate(top5):
 
        print(
            f"{i+1}. "
            f"Score={c['score']:.4f}   "
            f"Scale={c['scale']}   "
            f"Center=({c['x']:.2f},{c['y']:.2f})"
        )
 
    if best is None:
        best = top5[0]
    print("\nFINAL REFINED MATCH")
    print("----------------------------")
    print(f"Score : {best['score']:.4f}")
    print(f"Scale : {best['scale']}")
    print(f"Angle : {best.get('angle', 'N/A')}")
    print(f"Center: ({best['x']:.2f}, {best['y']:.2f})")
    
    output = cv2.cvtColor(search, cv2.COLOR_GRAY2BGR)
 
    cv2.rectangle(
        output,
        (best["left"], best["top"]),
        (
            best["left"] + best["width"],
            best["top"] + best["height"],
        ),
        (0, 0, 255),
        2,
    )
 
    cv2.circle(
        output,
        (int(best["x"]), int(best["y"])),
        4,
        (0, 255, 0),
        -1,
    )
 
    cv2.imwrite("prediction.png", output)
 
    print("\nPrediction image saved as prediction.png")
 
    print(f"Runtime : {runtime:.3f} sec")
 
    print(
        f"\nFinal Prediction : ({best['x']:.2f}, {best['y']:.2f})"
    )
 
    return best["x"], best["y"]
 
 
def main():
 
    parser = argparse.ArgumentParser()
 
    parser.add_argument(
        "--reference",
        required=True,
    )
 
    parser.add_argument(
        "--search",
        required=True,
    )
 
    args = parser.parse_args()
 
    x, y = predict(
        args.reference,
        args.search,
    )
 
    print(f"\nOUTPUT : {x:.2f},{y:.2f}")
 
 
if __name__ == "__main__":
    main()