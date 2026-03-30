import { Injectable, inject } from '@angular/core';
import { MatSnackBar } from '@angular/material/snack-bar';

const SUCCESS_DURATION_MS = 3000;
const ERROR_DURATION_MS = 5000;

@Injectable({ providedIn: 'root' })
export class NotificationService {
  private snackBar = inject(MatSnackBar);

  success(message: string): void {
    this.snackBar.open(message, 'OK', {
      duration: SUCCESS_DURATION_MS,
      panelClass: 'snackbar-success',
    });
  }

  error(message: string): void {
    this.snackBar.open(message, 'Dismiss', {
      duration: ERROR_DURATION_MS,
      panelClass: 'snackbar-error',
    });
  }
}
