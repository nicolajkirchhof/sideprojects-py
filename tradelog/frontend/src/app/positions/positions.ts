import { Component, OnInit, inject } from '@angular/core';
import { CommonModule } from '@angular/common';
import { MatTableModule } from '@angular/material/table';
import { PositionsService, Position } from './positions.service';
import { ContentArea } from '../shared/content-area/content-area';

@Component({
  selector: 'app-positions',
  standalone: true,
  imports: [CommonModule, MatTableModule, ContentArea],
  templateUrl: './positions.html',
  styleUrl: './positions.css'
})
export class Positions implements OnInit {
  private positionsService = inject(PositionsService);

  positions: Position[] = [];
  displayedColumns: string[] = [
    'id',
    'instrumentId',
    'contractId',
    'type',
    'opened',
    'expiry',
    'closed',
    'size',
    'strike',
    'cost',
    'close',
    'comission',
    'multiplier',
    'closeReason',
  ];

  ngOnInit(): void {
    this.positionsService.getPositions().subscribe({
      next: (data) => (this.positions = data ?? []),
      error: (err) => {
        console.error('Failed to load positions', err);
        this.positions = [];
      },
    });
  }
}
