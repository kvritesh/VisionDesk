# =============================================================================
# VisionDesk — Gesture_Controller.py
# =============================================================================
# CHANGE LOG (all modifications are annotated inline with [CHANGE: reason])
# 
#   FIX-1  : pyautogui.FAILSAFE re-enabled; FailSafeException caught in loop
#   FIX-2  : NameError in set_finger_state except-branch (dist1 → dist)
#   FIX-3  : Bare except: replaced with except ZeroDivisionError
#   FIX-4  : classify_hands bare except: replaced with except (IndexError, AttributeError)
#   FIX-5  : scrollHorizontal key combo corrected (Ctrl removed; Shift+Scroll is standard)
#   FIX-6  : changesystemvolume COM object no longer re-created every frame
#   FIX-7  : changesystembrightness reads brightness only once per call
#   FIX-8  : Dead attribute Controller.trial removed
#
#   FEAT-1 : CooldownAction — one-shot gesture mechanism with neutral-gate reset
#   FEAT-2 : Alt+Tab mapped to Gest.PINKY (major hand, one-shot)
#   FEAT-3 : Play/Pause mapped to Gest.RING  (major hand, one-shot)
#   FEAT-4 : Screenshot mapped to Gest.THUMB (major hand, one-shot)
#
#   UI-1   : FPS counter drawn on frame
#   UI-2   : Current gesture name drawn on frame
#   UI-3   : Current action label drawn on frame
#   UI-4   : Gesture help overlay toggled with 'H' key
# =============================================================================

import cv2
import mediapipe as mp
import pyautogui
import math
import time                          # [FEAT-1] needed for cooldown timestamps
import platform                      # [FEAT-4] screenshot hotkey differs by OS
from enum import IntEnum
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from google.protobuf.json_format import MessageToDict
import screen_brightness_control as sbcontrol
from screen_brightness_control import ScreenBrightnessError

# [FIX-1] FAILSAFE re-enabled so corner-of-screen acts as emergency kill switch.
# The main loop catches FailSafeException and releases any held mouse button
# before exiting cleanly.
pyautogui.FAILSAFE = True

mp_drawing = mp.solutions.drawing_utils
mp_hands   = mp.solutions.hands

# =============================================================================
# Gesture Encodings  (UNCHANGED — binary finger encoding preserved exactly)
# =============================================================================
class Gest(IntEnum):
    """Enum for mapping all hand gestures to binary numbers."""

    FIST             = 0
    PINKY            = 1
    RING             = 2
    MID              = 4
    LAST3            = 7
    INDEX            = 8
    FIRST2           = 12
    LAST4            = 15
    THUMB            = 16
    PALM             = 31

    # Extra Mappings
    V_GEST           = 33
    TWO_FINGER_CLOSED = 34
    PINCH_MAJOR      = 35
    PINCH_MINOR      = 36


# =============================================================================
# Multi-handedness Labels  (UNCHANGED)
# =============================================================================
class HLabel(IntEnum):
    MINOR = 0
    MAJOR = 1


# =============================================================================
# [FEAT-1]  CooldownAction — reusable one-shot gesture wrapper
# =============================================================================
class CooldownAction:
    """
    Wraps a callable so it fires at most once per gesture activation,
    and only resets after the hand returns to a neutral gesture (PALM)
    for at least `neutral_frames` consecutive frames.

    This prevents repeated firing while the user holds a gesture pose
    and makes each discrete action feel like a single button press.

    Parameters
    ----------
    action       : callable  — the OS action to execute (no args)
    label        : str       — human-readable name shown in the UI
    neutral_frames : int     — frames of PALM required to re-arm (default 8)
    """

    def __init__(self, action, label, neutral_frames=8):
        self._action        = action
        self.label          = label
        self._neutral_frames = neutral_frames
        self._armed         = True   # ready to fire
        self._neutral_count = 0      # frames seen as PALM since last fire

    def try_fire(self):
        """Fire the action if armed. Returns True if it fired."""
        if self._armed:
            self._action()
            self._armed = False
            self._neutral_count = 0
            return True
        return False

    def tick_neutral(self):
        """Call every frame when the current gesture IS neutral (PALM).
        Re-arms the action once enough neutral frames have accumulated."""
        self._neutral_count += 1
        if self._neutral_count >= self._neutral_frames:
            self._armed = True

    def reset_neutral(self):
        """Call every frame when the current gesture is NOT neutral."""
        self._neutral_count = 0


# =============================================================================
# HandRecog — Convert MediaPipe landmarks to gestures  (UNCHANGED except FIX-2/3)
# =============================================================================
class HandRecog:
    """
    Convert MediaPipe Landmarks to recognizable Gestures.
    """

    def __init__(self, hand_label):
        """
        Attributes
        ----------
        finger       : int  — current frame gesture encoding (binary)
        ori_gesture  : int  — debounced gesture being acted on
        prev_gesture : int  — gesture from previous frame
        frame_count  : int  — consecutive frames with same gesture
        hand_result  : obj  — MediaPipe landmark object
        hand_label   : int  — HLabel.MAJOR or HLabel.MINOR
        """
        self.finger       = 0
        self.ori_gesture  = Gest.PALM
        self.prev_gesture = Gest.PALM
        self.frame_count  = 0
        self.hand_result  = None
        self.hand_label   = hand_label

    def update_hand_result(self, hand_result):
        self.hand_result = hand_result

    def get_signed_dist(self, point):
        """Returns signed Euclidean distance between two landmarks."""
        sign = -1
        if self.hand_result.landmark[point[0]].y < self.hand_result.landmark[point[1]].y:
            sign = 1
        dist  = (self.hand_result.landmark[point[0]].x - self.hand_result.landmark[point[1]].x) ** 2
        dist += (self.hand_result.landmark[point[0]].y - self.hand_result.landmark[point[1]].y) ** 2
        return math.sqrt(dist) * sign

    def get_dist(self, point):
        """Returns Euclidean distance between two landmarks."""
        dist  = (self.hand_result.landmark[point[0]].x - self.hand_result.landmark[point[1]].x) ** 2
        dist += (self.hand_result.landmark[point[0]].y - self.hand_result.landmark[point[1]].y) ** 2
        return math.sqrt(dist)

    def get_dz(self, point):
        """Returns absolute z-axis difference between two landmarks."""
        return abs(self.hand_result.landmark[point[0]].z - self.hand_result.landmark[point[1]].z)

    def set_finger_state(self):
        """
        Sets self.finger by computing the open/closed ratio for each of the
        four non-thumb fingers.  Finger state: 1 = open, 0 = closed.

        UNCHANGED from original except:
          [FIX-2] except branch corrected: 'dist1' → 'dist' (was NameError)
          [FIX-3] bare except: → except ZeroDivisionError (explicit catch)
        """
        if self.hand_result is None:
            return

        points = [[8, 5, 0], [12, 9, 0], [16, 13, 0], [20, 17, 0]]
        self.finger = 0
        self.finger = self.finger | 0  # thumb always 0 (unchanged)

        for idx, point in enumerate(points):
            dist  = self.get_signed_dist(point[:2])
            dist2 = self.get_signed_dist(point[1:])

            try:
                ratio = round(dist / dist2, 1)
            except ZeroDivisionError:                    # [FIX-3] explicit type
                ratio = round(dist / 0.01, 1)           # [FIX-2] 'dist' not 'dist1'

            self.finger = self.finger << 1
            if ratio > 0.5:
                self.finger = self.finger | 1

    def get_gesture(self):
        """
        Returns debounced gesture (Gest enum value).
        A gesture must persist for >4 frames before it becomes ori_gesture.

        UNCHANGED from original.
        """
        if self.hand_result is None:
            return Gest.PALM

        current_gesture = Gest.PALM

        if self.finger in [Gest.LAST3, Gest.LAST4] and self.get_dist([8, 4]) < 0.05:
            if self.hand_label == HLabel.MINOR:
                current_gesture = Gest.PINCH_MINOR
            else:
                current_gesture = Gest.PINCH_MAJOR

        elif Gest.FIRST2 == self.finger:
            point = [[8, 12], [5, 9]]
            dist1 = self.get_dist(point[0])
            dist2 = self.get_dist(point[1])
            ratio = dist1 / dist2
            if ratio > 1.7:
                current_gesture = Gest.V_GEST
            else:
                if self.get_dz([8, 12]) < 0.1:
                    current_gesture = Gest.TWO_FINGER_CLOSED
                else:
                    current_gesture = Gest.MID

        else:
            current_gesture = self.finger

        if current_gesture == self.prev_gesture:
            self.frame_count += 1
        else:
            self.frame_count = 0

        self.prev_gesture = current_gesture

        if self.frame_count > 4:
            self.ori_gesture = current_gesture
        return self.ori_gesture


# =============================================================================
# Controller — Executes OS commands for each gesture
# =============================================================================
class Controller:
    """
    Executes OS commands according to detected gestures.

    Class-level attributes (static design preserved from original):
      tx_old, ty_old       : previous cursor position for dampening
      flag                 : True while in V_GEST (cursor-move mode)
      grabflag             : True while FIST drag is active
      pinchmajorflag       : True while PINCH_MAJOR is active
      pinchminorflag       : True while PINCH_MINOR is active
      pinch_threshold      : quantization step for pinch levels
      prev_hand            : previous hand position for dampening
      current_action_label : [UI-3] string shown in HUD as current action
      _volume_interface    : [FIX-6] cached COM audio object
      _cooldowns           : [FEAT-1] dict of gesture → CooldownAction
    """

    tx_old = 0
    ty_old = 0
    # [FIX-8] 'trial' removed — was declared but never used anywhere
    flag            = False
    grabflag        = False
    pinchmajorflag  = False
    pinchminorflag  = False
    pinchstartxcoord   = None
    pinchstartycoord   = None
    pinchdirectionflag = None
    prevpinchlv     = 0
    pinchlv         = 0
    framecount      = 0
    prev_hand       = None
    pinch_threshold = 0.3

    # [UI-3] Current action label for HUD display
    current_action_label = "Idle"

    # [FIX-6] Cache the COM audio interface so it is created once, not per-frame
    _volume_interface = None

    # -------------------------------------------------------------------------
    # [FIX-6]  Lazy-init the volume COM object exactly once
    # -------------------------------------------------------------------------
    @staticmethod
    def _get_volume_interface():
        """Returns cached IAudioEndpointVolume; creates it on first call."""
        if Controller._volume_interface is None:
            devices   = AudioUtilities.GetSpeakers()
            interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
            Controller._volume_interface = cast(interface, POINTER(IAudioEndpointVolume))
        return Controller._volume_interface

    # -------------------------------------------------------------------------
    # [FEAT-1]  One-shot CooldownAction table for discrete gestures
    # -------------------------------------------------------------------------
    # Populated after class body so lambda captures are resolved correctly.
    _cooldowns = {}

    # -------------------------------------------------------------------------
    # Pinch helpers  (UNCHANGED)
    # -------------------------------------------------------------------------
    def getpinchylv(hand_result):
        """Returns displacement from pinch-start along y-axis (scaled ×10)."""
        return round((Controller.pinchstartycoord - hand_result.landmark[8].y) * 10, 1)

    def getpinchxlv(hand_result):
        """Returns displacement from pinch-start along x-axis (scaled ×10)."""
        return round((hand_result.landmark[8].x - Controller.pinchstartxcoord) * 10, 1)

    # -------------------------------------------------------------------------
    # System-level actions  (brightness/volume fixed; scroll key-combo fixed)
    # -------------------------------------------------------------------------
    def changesystembrightness():
        """
        Adjusts system brightness by Controller.pinchlv / 50.

        [FIX-7] get_brightness() called only once per invocation (was called
        twice — once for the new level calculation, once as start= arg —
        creating a race condition).
        """
        try:
            current = sbcontrol.get_brightness(display=0)          # single read
            new_lv  = current / 100.0 + Controller.pinchlv / 50.0
            new_lv  = max(0.0, min(1.0, new_lv))
            sbcontrol.fade_brightness(int(100 * new_lv), start=current)  # reuse value
        except ScreenBrightnessError:
            print("Brightness control unavailable on this system.")

    def changesystemvolume():
        """
        Adjusts system volume by Controller.pinchlv / 50.

        [FIX-6] Uses cached COM interface instead of re-creating it every frame.
        """
        volume    = Controller._get_volume_interface()
        current   = volume.GetMasterVolumeLevelScalar()
        new_lv    = max(0.0, min(1.0, current + Controller.pinchlv / 50.0))
        volume.SetMasterVolumeLevelScalar(new_lv, None)

    def scrollVertical():
        """Scrolls the screen vertically.  UNCHANGED."""
        pyautogui.scroll(120 if Controller.pinchlv > 0.0 else -120)

    def scrollHorizontal():
        """
        Scrolls the screen horizontally.

        [FIX-5] Removed Ctrl modifier.  The original used Shift+Ctrl+Scroll
        which triggers zoom in most applications instead of horizontal scroll.
        The correct cross-application shortcut is Shift+Scroll only.
        """
        pyautogui.keyDown('shift')
        pyautogui.scroll(-120 if Controller.pinchlv > 0.0 else 120)
        pyautogui.keyUp('shift')

    # -------------------------------------------------------------------------
    # [FEAT-2/3/4]  One-shot action implementations
    # -------------------------------------------------------------------------
    def _do_alt_tab():
        """Fires Alt+Tab to switch windows."""
        pyautogui.hotkey('alt', 'tab')

    def _do_play_pause():
        """Fires the media Play/Pause key."""
        pyautogui.press('playpause')

    def _do_screenshot():
        """
        Captures a screenshot using the OS-native shortcut.
        Windows : Win+PrtSc  (saves to Pictures/Screenshots)
        macOS   : Cmd+Shift+3 (saves to Desktop)
        Linux   : PrtSc
        """
        os_name = platform.system()
        if os_name == 'Windows':
            pyautogui.hotkey('win', 'prtsc')
        elif os_name == 'Darwin':
            pyautogui.hotkey('command', 'shift', '3')
        else:
            pyautogui.press('prtsc')

    # -------------------------------------------------------------------------
    # Cursor position / dampening  (UNCHANGED)
    # -------------------------------------------------------------------------
    def get_position(hand_result):
        """
        Returns smoothed (x, y) cursor coordinates from hand landmark 9.
        Applies non-linear dampening: tiny movements are suppressed,
        large movements are passed through at ratio 2.1.

        UNCHANGED from original.
        """
        point    = 9
        position = [hand_result.landmark[point].x, hand_result.landmark[point].y]
        sx, sy   = pyautogui.size()
        x_old, y_old = pyautogui.position()
        x = int(position[0] * sx)
        y = int(position[1] * sy)

        if Controller.prev_hand is None:
            Controller.prev_hand = x, y

        delta_x = x - Controller.prev_hand[0]
        delta_y = y - Controller.prev_hand[1]
        distsq  = delta_x ** 2 + delta_y ** 2
        ratio   = 1
        Controller.prev_hand = [x, y]

        if distsq <= 25:
            ratio = 0
        elif distsq <= 900:
            ratio = 0.07 * (distsq ** 0.5)
        else:
            ratio = 2.1

        return (x_old + delta_x * ratio, y_old + delta_y * ratio)

    # -------------------------------------------------------------------------
    # Pinch gesture control  (UNCHANGED)
    # -------------------------------------------------------------------------
    def pinch_control_init(hand_result):
        """Initializes pinch-gesture state."""
        Controller.pinchstartxcoord = hand_result.landmark[8].x
        Controller.pinchstartycoord = hand_result.landmark[8].y
        Controller.pinchlv          = 0
        Controller.prevpinchlv      = 0
        Controller.framecount       = 0

    def pinch_control(hand_result, controlHorizontal, controlVertical):
        """
        Dispatches to controlHorizontal or controlVertical based on dominant
        axis of pinch movement.  Fires every 5 stable frames.

        UNCHANGED from original.
        """
        if Controller.framecount == 5:
            Controller.framecount = 0
            Controller.pinchlv    = Controller.prevpinchlv

            if Controller.pinchdirectionflag is True:
                controlHorizontal()
            elif Controller.pinchdirectionflag is False:
                controlVertical()

        lvx = Controller.getpinchxlv(hand_result)
        lvy = Controller.getpinchylv(hand_result)

        if abs(lvy) > abs(lvx) and abs(lvy) > Controller.pinch_threshold:
            Controller.pinchdirectionflag = False
            if abs(Controller.prevpinchlv - lvy) < Controller.pinch_threshold:
                Controller.framecount += 1
            else:
                Controller.prevpinchlv = lvy
                Controller.framecount  = 0

        elif abs(lvx) > Controller.pinch_threshold:
            Controller.pinchdirectionflag = True
            if abs(Controller.prevpinchlv - lvx) < Controller.pinch_threshold:
                Controller.framecount += 1
            else:
                Controller.prevpinchlv = lvx
                Controller.framecount  = 0

    # -------------------------------------------------------------------------
    # Main dispatcher
    # -------------------------------------------------------------------------
    def handle_controls(gesture, hand_result):
        """
        Dispatches OS actions based on the current gesture.

        Existing gesture→action mapping is 100% preserved.
        Three new one-shot gestures added at the bottom of the elif chain:
          Gest.PINKY  → Alt+Tab       [FEAT-2]
          Gest.RING   → Play/Pause    [FEAT-3]
          Gest.THUMB  → Screenshot    [FEAT-4]

        Each new gesture uses CooldownAction.try_fire() so it fires exactly
        once per activation and requires a return to PALM to re-arm.
        """
        x, y = None, None
        if gesture != Gest.PALM:
            x, y = Controller.get_position(hand_result)

        # -- flag resets (UNCHANGED) ------------------------------------------
        if gesture != Gest.FIST and Controller.grabflag:
            Controller.grabflag = False
            pyautogui.mouseUp(button="left")

        if gesture != Gest.PINCH_MAJOR and Controller.pinchmajorflag:
            Controller.pinchmajorflag = False

        if gesture != Gest.PINCH_MINOR and Controller.pinchminorflag:
            Controller.pinchminorflag = False

        # -- [FEAT-1] tick cooldown timers ------------------------------------
        # Neutral tick advances all one-shot re-arm counters when hand is idle.
        for gest_key, cooldown in Controller._cooldowns.items():
            if gesture == Gest.PALM:
                cooldown.tick_neutral()
            elif gesture != gest_key:
                cooldown.reset_neutral()

        # -- existing gestures (UNCHANGED logic) ------------------------------
        if gesture == Gest.V_GEST:
            Controller.flag = True
            Controller.current_action_label = "Move Cursor"
            pyautogui.moveTo(x, y, duration=0.1)

        elif gesture == Gest.FIST:
            if not Controller.grabflag:
                Controller.grabflag = True
                pyautogui.mouseDown(button="left")
            Controller.current_action_label = "Drag & Drop"
            pyautogui.moveTo(x, y, duration=0.1)

        elif gesture == Gest.MID and Controller.flag:
            pyautogui.click()
            Controller.flag = False
            Controller.current_action_label = "Left Click"

        elif gesture == Gest.INDEX and Controller.flag:
            pyautogui.click(button='right')
            Controller.flag = False
            Controller.current_action_label = "Right Click"

        elif gesture == Gest.TWO_FINGER_CLOSED and Controller.flag:
            pyautogui.doubleClick()
            Controller.flag = False
            Controller.current_action_label = "Double Click"

        elif gesture == Gest.PINCH_MINOR:
            if not Controller.pinchminorflag:
                Controller.pinch_control_init(hand_result)
                Controller.pinchminorflag = True
            Controller.current_action_label = "Scroll"
            Controller.pinch_control(hand_result,
                                     Controller.scrollHorizontal,
                                     Controller.scrollVertical)

        elif gesture == Gest.PINCH_MAJOR:
            if not Controller.pinchmajorflag:
                Controller.pinch_control_init(hand_result)
                Controller.pinchmajorflag = True
            Controller.current_action_label = "Brightness / Volume"
            Controller.pinch_control(hand_result,
                                     Controller.changesystembrightness,
                                     Controller.changesystemvolume)

        # -- [FEAT-2] Alt+Tab : Gest.PINKY (pinky finger only) ----------------
        elif gesture == Gest.PINKY:
            if Controller._cooldowns[Gest.PINKY].try_fire():
                Controller.current_action_label = "Alt+Tab"

        # -- [FEAT-3] Play/Pause : Gest.RING (ring finger only) ---------------
        elif gesture == Gest.RING:
            if Controller._cooldowns[Gest.RING].try_fire():
                Controller.current_action_label = "Play / Pause"

        # -- [FEAT-4] Screenshot : Gest.THUMB (thumb only — value 16) ---------
        # Note: thumb detection relies on the binary fallback path in get_gesture()
        # since set_finger_state() always sets thumb bit to 0.  Gest.THUMB = 16
        # is therefore only reachable if the upstream detection is extended.
        # For now this branch is wired and ready; a thumb-enable patch is the
        # only prerequisite to activate it.
        elif gesture == Gest.THUMB:
            if Controller._cooldowns[Gest.THUMB].try_fire():
                Controller.current_action_label = "Screenshot"

        elif gesture == Gest.PALM:
            Controller.current_action_label = "Idle"


# -- Initialise cooldown table after class body so lambdas resolve correctly --
# [FEAT-1/2/3/4]
Controller._cooldowns = {
    Gest.PINKY: CooldownAction(Controller._do_alt_tab,    "Alt+Tab",    neutral_frames=8),
    Gest.RING:  CooldownAction(Controller._do_play_pause, "Play/Pause", neutral_frames=8),
    Gest.THUMB: CooldownAction(Controller._do_screenshot, "Screenshot", neutral_frames=8),
}


# =============================================================================
# [UI-1/2/3/4]  HUD drawing helpers
# =============================================================================

# Colour palette — BGR
_C_GREEN   = (0,   220,  80)
_C_CYAN    = (0,   220, 220)
_C_YELLOW  = (30,  220, 255)
_C_WHITE   = (240, 240, 240)
_C_BLACK   = (0,     0,   0)
_C_OVERLAY = (20,   20,  20)   # semi-opaque overlay background tint
_C_RED     = (60,   60, 220)

_FONT      = cv2.FONT_HERSHEY_SIMPLEX
_FONT_BOLD = cv2.FONT_HERSHEY_DUPLEX

# Gesture cheat-sheet rows  (label, description)
_HELP_ROWS = [
    ("GESTURE CHEAT SHEET",          ""),
    ("V (index+mid spread)",         "Move Cursor"),
    ("Middle Finger",                "Left Click  (needs V first)"),
    ("Index Finger",                 "Right Click (needs V first)"),
    ("Two Fingers Closed",           "Double Click (needs V first)"),
    ("Fist",                         "Drag & Drop"),
    ("Pinch Major — horizontal",     "Brightness"),
    ("Pinch Major — vertical",       "Volume"),
    ("Pinch Minor — horizontal",     "Scroll H"),
    ("Pinch Minor — vertical",       "Scroll V"),
    ("Pinky Only",                   "Alt+Tab  [NEW]"),
    ("Ring Only",                    "Play / Pause  [NEW]"),
    ("Thumb Only*",                  "Screenshot  [NEW] (*pending)"),
    ("",                             ""),
    ("ENTER",                        "Exit      |     H — toggle help"),
]


def draw_hud(image, fps, gesture_name, action_label, show_help):
    """
    Draws the HUD elements onto the frame in-place.

    Draws:
      [UI-1] FPS counter        — top-right corner
      [UI-2] Gesture name       — bottom-left corner
      [UI-3] Action label       — below gesture name
      [UI-4] Help overlay       — full-frame overlay when show_help is True
    """
    h, w = image.shape[:2]

    # -- [UI-1] FPS counter (top-right) ---------------------------------------
    fps_text = f"FPS: {fps:5.1f}"
    (tw, th), _ = cv2.getTextSize(fps_text, _FONT, 0.55, 1)
    cv2.rectangle(image, (w - tw - 12, 6), (w - 4, th + 14), _C_BLACK, -1)
    cv2.putText(image, fps_text,
                (w - tw - 8, th + 10),
                _FONT, 0.55, _C_CYAN, 1, cv2.LINE_AA)

    # -- [UI-2] Gesture name (bottom-left) ------------------------------------
    gest_text = f"Gesture: {gesture_name}"
    cv2.putText(image, gest_text,
                (10, h - 40),
                _FONT_BOLD, 0.62, _C_BLACK, 4, cv2.LINE_AA)   # shadow
    cv2.putText(image, gest_text,
                (10, h - 40),
                _FONT_BOLD, 0.62, _C_GREEN, 1, cv2.LINE_AA)

    # -- [UI-3] Action label (bottom-left, below gesture) --------------------
    act_text = f"Action : {action_label}"
    cv2.putText(image, act_text,
                (10, h - 14),
                _FONT_BOLD, 0.62, _C_BLACK, 4, cv2.LINE_AA)   # shadow
    cv2.putText(image, act_text,
                (10, h - 14),
                _FONT_BOLD, 0.62, _C_YELLOW, 1, cv2.LINE_AA)

    # -- [UI-4] Help overlay --------------------------------------------------
    if show_help:
        # Semi-transparent dark rectangle
        overlay = image.copy()
        cv2.rectangle(overlay, (30, 20), (w - 30, h - 20), _C_OVERLAY, -1)
        cv2.addWeighted(overlay, 0.78, image, 0.22, 0, image)

        row_h    = 26
        start_y  = 52
        for i, (lbl, desc) in enumerate(_HELP_ROWS):
            y = start_y + i * row_h

            if i == 0:
                # Title row
                cv2.putText(image, lbl,
                            (w // 2 - 130, y),
                            _FONT_BOLD, 0.72, _C_CYAN, 1, cv2.LINE_AA)
            elif lbl == "":
                pass  # spacer row
            else:
                cv2.putText(image, lbl,
                            (50, y),
                            _FONT, 0.50, _C_WHITE, 1, cv2.LINE_AA)
                cv2.putText(image, f"→  {desc}",
                            (w // 2 - 20, y),
                            _FONT, 0.50, _C_GREEN, 1, cv2.LINE_AA)

    return image


# =============================================================================
# GestureController — main class  (camera loop updated for HUD + FailSafe fix)
# =============================================================================

'''
----------------------------------------  Main Class  ----------------------------------------
    Entry point of Gesture Controller
'''


class GestureController:
    """
    Handles camera capture, landmark extraction via MediaPipe, and
    dispatches to HandRecog and Controller.

    Attributes
    ----------
    gc_mode    : int   — 1 = running, 0 = stopped
    cap        : obj   — cv2.VideoCapture handle
    CAM_HEIGHT : float — camera frame height in pixels
    CAM_WIDTH  : float — camera frame width in pixels
    hr_major   : obj   — landmark result for major hand
    hr_minor   : obj   — landmark result for minor hand
    dom_hand   : bool  — True = right hand is dominant
    """
    gc_mode    = 0
    cap        = None
    CAM_HEIGHT = None
    CAM_WIDTH  = None
    hr_major   = None
    hr_minor   = None
    dom_hand   = True

    def __init__(self):
        """Initialises camera and sets gc_mode to running."""
        GestureController.gc_mode    = 1
        GestureController.cap        = cv2.VideoCapture(0)
        GestureController.CAM_HEIGHT = GestureController.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        GestureController.CAM_WIDTH  = GestureController.cap.get(cv2.CAP_PROP_FRAME_WIDTH)

    def classify_hands(results):
        """
        Assigns hr_major / hr_minor from MediaPipe multi-hand results.

        [FIX-4] Replaced bare except: with except (IndexError, AttributeError)
        so genuine programming errors (e.g. wrong key name) are not silently
        swallowed.  IndexError covers missing hand slots; AttributeError
        covers unexpected MediaPipe API shape changes.

        Logic UNCHANGED from original.
        """
        left, right = None, None

        try:
            handedness_dict = MessageToDict(results.multi_handedness[0])
            if handedness_dict['classification'][0]['label'] == 'Right':
                right = results.multi_hand_landmarks[0]
            else:
                left = results.multi_hand_landmarks[0]
        except (IndexError, AttributeError):          # [FIX-4]
            pass

        try:
            handedness_dict = MessageToDict(results.multi_handedness[1])
            if handedness_dict['classification'][0]['label'] == 'Right':
                right = results.multi_hand_landmarks[1]
            else:
                left = results.multi_hand_landmarks[1]
        except (IndexError, AttributeError):          # [FIX-4]
            pass

        if GestureController.dom_hand:
            GestureController.hr_major = right
            GestureController.hr_minor = left
        else:
            GestureController.hr_major = left
            GestureController.hr_minor = right

    def start(self):
        """
        Main capture loop.

        Changes from original:
          [FIX-1]  Catches pyautogui.FailSafeException to release mouse safely
          [UI-1]   FPS measured with time.time() deltas
          [UI-2/3] gesture_name / action_label passed to draw_hud()
          [UI-4]   'H' key toggles help overlay; show_help state maintained
        """
        handmajor   = HandRecog(HLabel.MAJOR)
        handminor   = HandRecog(HLabel.MINOR)
        show_help   = False          # [UI-4] help overlay toggle state
        prev_time   = time.time()    # [UI-1] for FPS calculation
        fps         = 0.0
        gest_name   = Gest.PALM      # current debounced gesture for display

        with mp_hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as hands:

            while GestureController.cap.isOpened() and GestureController.gc_mode:
                success, image = GestureController.cap.read()

                if not success:
                    print("Ignoring empty camera frame.")
                    continue

                # -- [UI-1] FPS calculation ------------------------------------
                now      = time.time()
                fps      = 1.0 / max(now - prev_time, 1e-6)
                prev_time = now

                # -- MediaPipe processing (UNCHANGED) --------------------------
                image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                results = hands.process(image)

                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                if results.multi_hand_landmarks:
                    GestureController.classify_hands(results)
                    handmajor.update_hand_result(GestureController.hr_major)
                    handminor.update_hand_result(GestureController.hr_minor)

                    handmajor.set_finger_state()
                    handminor.set_finger_state()
                    gest_name = handminor.get_gesture()

                    if gest_name == Gest.PINCH_MINOR:
                        Controller.handle_controls(gest_name, handminor.hand_result)
                    else:
                        gest_name = handmajor.get_gesture()
                        Controller.handle_controls(gest_name, handmajor.hand_result)

                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(
                            image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                else:
                    Controller.prev_hand = None
                    gest_name = Gest.PALM

                # -- [UI-1/2/3/4] Draw HUD onto frame -------------------------
                # Resolve gesture enum to a display-friendly string
                try:
                    gest_display = Gest(gest_name).name.replace('_', ' ').title()
                except ValueError:
                    gest_display = str(gest_name)

                image = draw_hud(
                    image,
                    fps,
                    gest_display,
                    Controller.current_action_label,
                    show_help
                )

                cv2.imshow('VisionDesk — Gesture Controller', image)

                # -- Key handling ---------------------------------------------
                key = cv2.waitKey(5) & 0xFF
                if key == 13:                   # Enter → exit
                    GestureController.gc_mode = 0
                    break
                elif key == ord('h') or key == ord('H'):   # [UI-4] toggle help
                    show_help = not show_help

        # [FIX-1] Release mouse before closing in case a drag was in progress
        try:
            if Controller.grabflag:
                pyautogui.mouseUp(button="left")
        except Exception:
            pass

        GestureController.cap.release()
        cv2.destroyAllWindows()


# =============================================================================
# Entry point
# =============================================================================
try:
    gc1 = GestureController()
    gc1.start()
except pyautogui.FailSafeException:
    # [FIX-1] Cursor reached a screen corner — release mouse and exit cleanly
    print("[VisionDesk] FailSafe triggered — cursor moved to screen corner.")
    try:
        if Controller.grabflag:
            pyautogui.mouseUp(button="left")
    except Exception:
        pass
    if GestureController.cap is not None:
        GestureController.cap.release()
    cv2.destroyAllWindows()
