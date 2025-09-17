import { Component, OnInit, inject, signal } from '@angular/core';
import { RouterOutlet } from '@angular/router';
import { HttpClient } from '@angular/common/http';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [RouterOutlet],
  templateUrl: './app.html',
  styleUrl: './app.css'
})
export class App implements OnInit {
  private http = inject(HttpClient);
  title = 'frontend-app';
  message = signal('');

  ngOnInit(): void {
    this.http.get<{ message: string }>('http://127.0.0.1:5000/').subscribe(data => {
      this.message.set(data.message);
    });
  }
}