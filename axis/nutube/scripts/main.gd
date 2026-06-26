extends Control

## nuTube main screen.
##
## Seeds the local index from a few hardcoded YouTube videos, then renders
## whatever LocalIndex ranks. There is no remote algorithm: titles/thumbnails
## come from YouTube's key-free endpoints, but the ranking is all local.
##
## The index and its thumbnails are cached to `user://` (see LocalIndex and
## YouTubeSource), so only the first launch pays the network cost - later
## launches render straight from disk and fetch nothing.

const SEED_VIDEOS := [
	"https://youtu.be/DvFfX4Ih_XM",
	"https://youtu.be/hxE7aW8S3rc",
	"https://youtu.be/x9kmvIPrcbE",
]

@onready var search_bar: LineEdit = $Margin/VBox/SearchRow/SearchBar
@onready var search_button: Button = $Margin/VBox/SearchRow/SearchButton
@onready var feed_list: VBoxContainer = $Margin/VBox/Feed/FeedList

var _youtube: YouTubeSource

# Detail overlay, built once in `_ready`.
var _detail: CanvasLayer
var _detail_thumb: TextureRect
var _detail_title: Label
var _detail_author: Label
var _detail_url: String = ""


func _ready() -> void:
	_youtube = YouTubeSource.new()
	add_child(_youtube)
	_youtube.item_resolved.connect(_on_item_resolved)

	search_button.pressed.connect(_on_explore)
	search_bar.text_submitted.connect(func(_text): _on_explore())

	_build_detail()

	# Render whatever is already cached first, so a warm launch is instant,
	# then only hit the network for videos we don't fully have yet.
	_render(LocalIndex.recommend("", 10))
	for url in SEED_VIDEOS:
		if LocalIndex.needs_fetch(_youtube.id_from_url(url)):
			_youtube.resolve(url)


func _on_item_resolved(item: Dictionary) -> void:
	LocalIndex.upsert(item)
	LocalIndex.save_index()
	# Re-rank with whatever is in the search box right now.
	_on_explore()


func _on_explore() -> void:
	var query := search_bar.text.strip_edges()
	_render(LocalIndex.recommend(query, 10))


func _render(items: Array) -> void:
	for child in feed_list.get_children():
		child.queue_free()
	if items.is_empty():
		feed_list.add_child(_muted("Fetching videos..."))
		return
	for item in items:
		feed_list.add_child(_make_card(item))


func _make_card(item: Dictionary) -> Control:
	var card := PanelContainer.new()
	card.gui_input.connect(func(event): _on_card_input(event, item))

	var box := VBoxContainer.new()
	box.mouse_filter = Control.MOUSE_FILTER_IGNORE
	box.add_theme_constant_override("separation", 8)
	card.add_child(box)

	var thumbnail: Texture2D = item.get("thumbnail")
	if thumbnail:
		box.add_child(_thumbnail_rect(thumbnail, 200))

	var title_label := Label.new()
	title_label.text = str(item.get("title", "Untitled"))
	title_label.add_theme_font_size_override("font_size", 22)
	title_label.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	title_label.mouse_filter = Control.MOUSE_FILTER_IGNORE
	box.add_child(title_label)

	var author := str(item.get("author", ""))
	if author != "":
		box.add_child(_muted(author))
	var reason := str(item.get("reason", ""))
	if reason != "" and reason != "From your local index":
		box.add_child(_muted(reason))
	return card


func _on_card_input(event: InputEvent, item: Dictionary) -> void:
	if event is InputEventMouseButton and event.pressed and event.button_index == MOUSE_BUTTON_LEFT:
		_open_detail(item)


# --- detail overlay -------------------------------------------------------
# A deliberate two-step: tapping a card opens this, and only an explicit
# "Watch" tap hands off to YouTube. Stops accidental scroll-taps from spawning
# a pile of background browser tabs, and is where an embedded player will live.


func _build_detail() -> void:
	_detail = CanvasLayer.new()
	_detail.layer = 10
	add_child(_detail)

	var scrim := ColorRect.new()
	scrim.color = Color(0, 0, 0, 0.85)
	scrim.set_anchors_and_offsets_preset(Control.PRESET_FULL_RECT)
	scrim.gui_input.connect(_on_scrim_input)
	_detail.add_child(scrim)

	var margin := MarginContainer.new()
	margin.set_anchors_and_offsets_preset(Control.PRESET_FULL_RECT)
	margin.mouse_filter = Control.MOUSE_FILTER_IGNORE
	for side in ["left", "top", "right", "bottom"]:
		margin.add_theme_constant_override("margin_" + side, 24)
	scrim.add_child(margin)

	var box := VBoxContainer.new()
	box.mouse_filter = Control.MOUSE_FILTER_IGNORE
	box.alignment = BoxContainer.ALIGNMENT_CENTER
	box.add_theme_constant_override("separation", 16)
	margin.add_child(box)

	_detail_thumb = _thumbnail_rect(null, 260)
	box.add_child(_detail_thumb)

	_detail_title = Label.new()
	_detail_title.add_theme_font_size_override("font_size", 26)
	_detail_title.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	_detail_title.mouse_filter = Control.MOUSE_FILTER_IGNORE
	box.add_child(_detail_title)

	_detail_author = _muted("")
	box.add_child(_detail_author)

	var buttons := HBoxContainer.new()
	buttons.mouse_filter = Control.MOUSE_FILTER_IGNORE
	buttons.add_theme_constant_override("separation", 12)
	box.add_child(buttons)

	var watch := Button.new()
	watch.text = "Watch on YouTube"
	watch.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	watch.pressed.connect(func(): if _detail_url != "": OS.shell_open(_detail_url))
	buttons.add_child(watch)

	var back := Button.new()
	back.text = "Back"
	back.pressed.connect(_close_detail)
	buttons.add_child(back)

	_detail.hide()


func _open_detail(item: Dictionary) -> void:
	_detail_url = str(item.get("url", ""))
	_detail_thumb.texture = item.get("thumbnail")
	_detail_thumb.visible = _detail_thumb.texture != null
	_detail_title.text = str(item.get("title", "Untitled"))
	var author := str(item.get("author", ""))
	_detail_author.text = author
	_detail_author.visible = author != ""
	_detail.show()


func _close_detail() -> void:
	_detail.hide()


func _on_scrim_input(event: InputEvent) -> void:
	# Tap on the dimmed background (outside the buttons) closes the overlay.
	if event is InputEventMouseButton and event.pressed and event.button_index == MOUSE_BUTTON_LEFT:
		_close_detail()


# --- small builders -------------------------------------------------------


func _thumbnail_rect(texture: Texture2D, min_height: int) -> TextureRect:
	var rect := TextureRect.new()
	rect.texture = texture
	rect.expand_mode = TextureRect.EXPAND_IGNORE_SIZE
	rect.stretch_mode = TextureRect.STRETCH_KEEP_ASPECT_COVERED
	rect.custom_minimum_size = Vector2(0, min_height)
	rect.clip_contents = true
	rect.mouse_filter = Control.MOUSE_FILTER_IGNORE
	return rect


func _muted(text: String) -> Label:
	var label := Label.new()
	label.text = text
	label.add_theme_color_override("font_color", Color(0.6, 0.65, 0.72))
	label.mouse_filter = Control.MOUSE_FILTER_IGNORE
	return label
