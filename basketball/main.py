import argparse
import os
import time
from utils  import read_frames , write_video , tracker_hallosination
from infering import Inference
from draw import Draw
from visualizations import draw_mini_court
from scoring_system import find_events  , find_stats

def main(track_only=False):
    os.makedirs(os.path.join(os.getcwd(), 'presaved', 'smoke_test'), exist_ok=True)

    frames = read_frames(os.path.join(os.getcwd() , 'inputs/ja_morant-block.mp4'))
    print(f'[smoke-test] frames leídos: {len(frames)}')
    inference_class= Inference(frames )

    t0 = time.time()
    detections = inference_class.player_ball_detectioin('basketball-player-detection-3-ycjdo/4' , presaved=False  , path = os.path.join(os.getcwd() , 'presaved/smoke_test/player_ball.pkl'))
    t1 = time.time()
    print(f'[smoke-test] player_ball_detectioin: {t1 - t0:.1f}s ({(t1 - t0)/len(frames):.2f}s/frame)')

    court_detections = inference_class.court_keypoints('basketball-court-detection-2/14' , presaved=False , path=os.path.join(os.getcwd() , 'presaved/smoke_test/court.pkl'))
    t2 = time.time()
    print(f'[smoke-test] court_keypoints: {t2 - t1:.1f}s ({(t2 - t1)/len(frames):.2f}s/frame)')

    teams = inference_class.one_shot_classification(detections , presaved=False , path = os.path.join(os.getcwd() , 'presaved/smoke_test/teams.pkl'))
    t3 = time.time()
    print(f'[smoke-test] one_shot_classification: {t3 - t2:.1f}s ({(t3 - t2)/len(frames):.2f}s/frame)')

    drawer =Draw()
    annotated_frames, corrected_team_classifications =drawer.draw_players_ball(
        detections , teams , frames , show_frame_counter=not track_only )
    annotated_frames = drawer.draw_key_points(court_detections , annotated_frames)

    if not track_only :
        events = find_events(detections)
        annotated_frames  , stats , foot_prints , cum= draw_mini_court(annotated_frames , detections , court_detections ,corrected_team_classifications , events )

    write_video(os.path.join(os.getcwd() , 'outputs/out_moran-block.mp4') , annotated_frames)
    print(f'[smoke-test] total inferencia: {t3 - t0:.1f}s — vídeo en outputs/out_moran-block.mp4')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--track-only', action='store_true',
                         help='Solo dibuja tracking de jugadores/balón y keypoints de cancha: '
                              'sin tabla de stats, footprints, heatmaps, mapa dinámico, '
                              'game momentum ni el contador de frame de debug.')
    args = parser.parse_args()
    main(track_only=args.track_only)
