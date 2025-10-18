import { Component, Input } from '@angular/core';
import { CommonModule } from '@angular/common';

@Component({
  selector: 'app-content-area',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './content-area.html',
  styleUrl: './content-area.css'
})
export class ContentArea {
  @Input() showRightSidebar = false;
  @Input() sidebarWidth: string = '320px';
  @Input() title?: string;
}
