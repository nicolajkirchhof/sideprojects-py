import { Component, computed, input } from '@angular/core';

import { AngularSplitModule } from 'angular-split';

@Component({
  selector: 'app-content-area',
  standalone: true,
  imports: [AngularSplitModule],
  templateUrl: './content-area.html',
  host: {class: 'flex flex-col flex-1'},
})
export class ContentArea {
  showRightSidebar = input(false);
  /** Initial sidebar width in pixels — matches the form max-width (420px) + padding. */
  sidebarWidth = input(460);
  sidebarWidthPx = computed(() => this.sidebarWidth());
  title = input<string>();
}
