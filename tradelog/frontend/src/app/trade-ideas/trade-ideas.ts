import { Component, OnInit, inject } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { MatTableModule } from '@angular/material/table';
import { DatePipe } from '@angular/common';
import { ContentArea } from '../shared/content-area/content-area';

export interface TradeIdea {
  id: number;
  date: string;
  rich_text_notes: string;
}

@Component({
  selector: 'app-trade-ideas',
  standalone: true,
  imports: [MatTableModule, DatePipe, ContentArea],
  templateUrl: './trade-ideas.html',
  styleUrl: './trade-ideas.css'
})
export class TradeIdeas implements OnInit {
  private http = inject(HttpClient);
  tradeIdeas: TradeIdea[] = [];
  displayedColumns: string[] = ['id', 'date', 'rich_text_notes'];

  ngOnInit(): void {
    this.http.get<TradeIdea[]>('http://127.0.0.1:5000/trade-ideas').subscribe(data => {
      this.tradeIdeas = data;
    });
  }
}
