import tkinter as tk
from datetime import datetime, timedelta
import winsound  # For the beep sound on Windows


def calculate_time_to_next_5th_minute():
  """Calculate the number of seconds until the next 5-minute mark."""
  now = datetime.now()
  next_5_minute = (now + timedelta(minutes=5 - now.minute % 5)).replace(second=0, microsecond=0)
  return (next_5_minute - now).seconds


def update_timer():
  """Update the countdown timer and dynamically change colors."""
  global remaining_time_seconds

  # Calculate minutes and seconds
  minutes, seconds = divmod(remaining_time_seconds, 60)
  timer_label.config(text=f"{minutes:02}:{seconds:02}")

  # Change text color based on time left
  if remaining_time_seconds > 30:
    timer_label.config(fg="#228B22")  # Muted Green
  else:
    timer_label.config(fg="#B22222")  # Muted Red

  # When the timer reaches zero, beep and reset
  if remaining_time_seconds <= 0:
    if sound_enabled:
      winsound.Beep(1000, 500)  # Beep sound (frequency: 1000Hz, duration: 500ms)
    remaining_time_seconds = calculate_time_to_next_5th_minute()  # Reset to next 5-min mark
  else:
    remaining_time_seconds -= 1  # Decrement the timer

  # Schedule this function to run again in 1 second
  root.after(1000, update_timer)


def toggle_sound():
  """Toggle the sound on/off and update the loudspeaker icon."""
  global sound_enabled

  sound_enabled = not sound_enabled  # Toggle the sound state
  if sound_enabled:
    sound_button.config(text="🔊")  # Loudspeaker ON
  else:
    sound_button.config(text="🔇")  # Loudspeaker OFF


# Initialize Timer
remaining_time_seconds = calculate_time_to_next_5th_minute()  # Time left to the next 5-min mark
sound_enabled = True  # Sound is enabled by default

# Create GUI
root = tk.Tk()
root.title("Always Counting Timer")

# Create a Frame for Timer + Sound button
main_frame = tk.Frame(root, bg="white", padx=10, pady=10)
main_frame.pack(fill="x")

# Create the timer display label
timer_label = tk.Label(main_frame, text="00:00", font=("Helvetica", 48), fg="#228B22", bg="white")
timer_label.pack(side="left", padx=10)

# Create the loudspeaker toggle button in the top-right corner
sound_button = tk.Button(main_frame, text="🔊", font=("Helvetica", 14), command=toggle_sound, bg="white", relief="flat")
sound_button.pack(side="right", padx=10)

# Start the timer automatically
update_timer()

# Run the Tkinter main loop
root.mainloop()
