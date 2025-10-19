import { Component, Input } from '@angular/core';
import { CommonModule } from '@angular/common';
import { AngularSplitModule } from 'angular-split';

@Component({
  selector: 'app-content-area',
  standalone: true,
  imports: [CommonModule, AngularSplitModule],
  templateUrl: './content-area.html',
  host: {class: 'flex flex-col flex-1'},
})
export class ContentArea {
  @Input() showRightSidebar = false;
  @Input() sidebarWidth: string = '320px';
  @Input() title?: string;
}
