import chess
import logging
from typing import Optional, Dict, Any, Type

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("Languages")

class LanguageBase:
    """
    Base class for all language implementations providing common functionality
    and default values for chess commentary.
    """
    
    def __init__(self):
        # Game status messages
        self.game_started = "Game started"
        self.move_failed = "Move registration failed. Please redo your move."
        self.game_over = "Game over"
        self.white_wins = "White wins"
        self.black_wins = "Black wins"
        self.draw = "Draw"
        self.game_over_resignation_or_draw = "Game ended by resignation or draw"
        
        # Piece movement modifiers
        self.captures = "captures"
        self.moves_to = "moves to"
        self.promotion_text = "promotion to"
        
        # Check and checkmate
        self.check = "check"
        self.checkmate = "checkmate"
        
        # Castling
        self.castling_kingside = "kingside castling"
        self.castling_queenside = "queenside castling"
        
    def name(self, piece_type: int) -> str:
        """
        Get the name of a chess piece in the current language.
        
        Args:
            piece_type: Chess piece type constant from python-chess
            
        Returns:
            Name of the piece in the current language
            
        Raises:
            ValueError: If piece_type is invalid
        """
        piece_names = {
            chess.PAWN: "pawn",
            chess.KNIGHT: "knight",
            chess.BISHOP: "bishop",
            chess.ROOK: "rook",
            chess.QUEEN: "queen",
            chess.KING: "king"
        }
        
        if piece_type not in piece_names:
            logger.error(f"Invalid piece type: {piece_type}")
            raise ValueError(f"Invalid piece type: {piece_type}")
            
        return piece_names[piece_type]
        
    def comment(self, board: chess.Board, move: chess.Move) -> str:
        """
        Generate a spoken commentary for a chess move.
        
        Args:
            board: Current chess board state (after the move)
            move: The move to comment on
            
        Returns:
            Spoken commentary text
            
        Raises:
            ValueError: If the move is invalid for the given board
        """
        try:
            # Check for special states, need to pop move to check properly
            check_status = ""
            if board.is_checkmate():
                check_status = f" {self.checkmate}"
            elif board.is_check():
                check_status = f" {self.check}"
                
            # Revert the board to check castling and movement
            board.pop()
            
            # Check for castling moves
            if board.is_kingside_castling(move):
                board.push(move)
                return f"{self.castling_kingside}{check_status}"
                
            if board.is_queenside_castling(move):
                board.push(move)
                return f"{self.castling_queenside}{check_status}"
            
            # Get piece information and squares
            piece = board.piece_at(move.from_square)
            if piece is None:
                logger.error(f"No piece at square {chess.square_name(move.from_square)}")
                board.push(move)  # Restore board state
                return "Invalid move"
                
            from_square = chess.square_name(move.from_square)
            to_square = chess.square_name(move.to_square)
            is_capture = board.is_capture(move)
            
            # Check for promotion
            promotion_text = ""
            if move.promotion:
                promotion_text = f" {self.promotion_text} {self.name(move.promotion)}"
            
            # Make the move
            board.push(move)
            
            # Build the commentary
            comment = f"{self.name(piece.piece_type)} {from_square}"
            comment += f" {self.captures}" if is_capture else f" {self.moves_to}"
            comment += f" {to_square}{promotion_text}{check_status}"
            
            return comment
            
        except chess.IllegalMoveError:
            logger.error(f"Illegal move: {move.uci()}")
            raise ValueError(f"Illegal move: {move.uci()}")
            
        except Exception as e:
            logger.error(f"Error generating commentary: {e}")
            # Ensure board state is preserved even if commentary fails
            if board.move_stack and board.peek() != move:
                try:
                    board.push(move)
                except chess.IllegalMoveError:
                    pass
            return "Move made"


class English(LanguageBase):
    """English language for chess commentary."""
    
    def __init__(self):
        super().__init__()
        self.game_started = "Game started"
        self.move_failed = "Move registration failed. Please redo your move."
        self.game_over = "Game over"
        self.white_wins = "White wins"
        self.black_wins = "Black wins"
        self.draw = "It's a draw"
        self.game_over_resignation_or_draw = "Game ended by resignation or draw"
        
        # Piece movement modifiers
        self.captures = "takes"
        self.moves_to = "to"
        self.promotion_text = "promotion to"
        
        # Check and checkmate
        self.check = "check"
        self.checkmate = "checkmate"
        
        # Castling
        self.castling_kingside = "castling short"
        self.castling_queenside = "castling long"


class German(LanguageBase):
    """German language for chess commentary."""
    
    def __init__(self):
        super().__init__()
        self.game_started = "Das Spiel hat gestartet."
        self.move_failed = "Der Zug ist ungültig, bitte wiederholen."
        self.game_over = "Spiel beendet"
        self.white_wins = "Weiß gewinnt"
        self.black_wins = "Schwarz gewinnt"
        self.draw = "Remis"
        self.game_over_resignation_or_draw = "Spiel durch Aufgabe oder Remis beendet"
        
        # Piece movement modifiers
        self.captures = "schlägt"
        self.moves_to = "nach"
        self.promotion_text = "Umwandlung in"
        
        # Check and checkmate
        self.check = "Schach"
        self.checkmate = "Schachmatt"
        
        # Castling
        self.castling_kingside = "kurze Rochade"
        self.castling_queenside = "lange Rochade"
    
    def name(self, piece_type: int) -> str:
        """Get German name for chess piece."""
        piece_names = {
            chess.PAWN: "Bauer",
            chess.KNIGHT: "Springer",
            chess.BISHOP: "Läufer",
            chess.ROOK: "Turm",
            chess.QUEEN: "Dame",
            chess.KING: "König"
        }
        
        if piece_type not in piece_names:
            logger.error(f"Invalid piece type: {piece_type}")
            return "unbekannt"
            
        return piece_names[piece_type]


class Russian(LanguageBase):
    """Russian language for chess commentary."""
    
    def __init__(self):
        super().__init__()
        self.game_started = "игра началась"
        self.move_failed = "Ошибка регистрации хода. Пожалуйста, повторите свой ход"
        self.game_over = "игра окончена"
        self.white_wins = "белые выиграли"
        self.black_wins = "черные выиграли"
        self.draw = "ничья"
        self.game_over_resignation_or_draw = "Игра закончилась отставкой или ничьей"
        
        # Piece movement modifiers
        self.captures = "бьёт"
        self.moves_to = ""  # Empty in Russian
        self.promotion_text = "превращение в"
        
        # Check and checkmate
        self.check = "шах"
        self.checkmate = "шах и мат"
        
        # Castling
        self.castling_kingside = "короткая рокировка"
        self.castling_queenside = "длинная рокировка"
    
    def name(self, piece_type: int) -> str:
        """Get Russian name for chess piece."""
        piece_names = {
            chess.PAWN: "пешка",
            chess.KNIGHT: "конь",
            chess.BISHOP: "слон",
            chess.ROOK: "ладья",
            chess.QUEEN: "ферзь",
            chess.KING: "король"
        }
        
        if piece_type not in piece_names:
            logger.error(f"Invalid piece type: {piece_type}")
            return "неизвестно"
            
        return piece_names[piece_type]


class Turkish(LanguageBase):
    """Turkish language for chess commentary."""
    
    def __init__(self):
        super().__init__()
        self.game_started = "Oyun başladı."
        self.move_failed = "Hamle geçersiz. Lütfen hamlenizi yeniden yapın."
        self.game_over = "Oyun bitti"
        self.white_wins = "Beyaz kazandı"
        self.black_wins = "Siyah kazandı"
        self.draw = "Berabere"
        self.game_over_resignation_or_draw = "Oyun terk veya beraberlik ile sonuçlandı"
    
    def name(self, piece_type: int) -> str:
        """Get Turkish name for chess piece."""
        piece_names = {
            chess.PAWN: "piyon",
            chess.KNIGHT: "at",
            chess.BISHOP: "fil",
            chess.ROOK: "kale",
            chess.QUEEN: "vezir",
            chess.KING: "şah"
        }
        
        if piece_type not in piece_names:
            logger.error(f"Invalid piece type: {piece_type}")
            return "bilinmeyen"
            
        return piece_names[piece_type]
    
    def capture_suffix(self, to_square: str) -> str:
        """Get the correct Turkish suffix for capture based on square name."""
        try:
            if to_square[-1] in "158":
                return "i"
            elif to_square[-1] in "27":
                return "yi"
            elif to_square[-1] in "34":
                return "ü"
            else:  # 6
                return "yı"
        except IndexError:
            logger.error(f"Invalid square name for capture suffix: {to_square}")
            return ""
    
    def from_suffix(self, from_square: str) -> str:
        """Get the correct Turkish suffix for movement origin based on square name."""
        try:
            if from_square[-1] in "1278":
                return "den"
            elif from_square[-1] in "345":
                return "ten"
            else:  # 6
                return "dan"
        except IndexError:
            logger.error(f"Invalid square name for from suffix: {from_square}")
            return ""
    
    def to_suffix(self, to_square: str) -> str:
        """Get the correct Turkish suffix for movement destination based on square name."""
        try:
            if to_square[-1] in "13458":
                return "e"
            elif to_square[-1] in "27":
                return "ye"
            else:  # 6
                return "ya"
        except IndexError:
            logger.error(f"Invalid square name for to suffix: {to_square}")
            return ""
    
    def comment(self, board: chess.Board, move: chess.Move) -> str:
        """Generate Turkish commentary for a chess move with proper suffixes."""
        try:
            # Check for special states
            check_status = ""
            if board.is_checkmate():
                check_status = " şahmat"
            elif board.is_check():
                check_status = " şah"
                
            # Revert the board to check castling and movement
            board.pop()
            
            # Check for castling moves
            if board.is_kingside_castling(move):
                board.push(move)
                return f"kısa rok{check_status}"
                
            if board.is_queenside_castling(move):
                board.push(move)
                return f"uzun rok{check_status}"
            
            # Get piece information and squares
            piece = board.piece_at(move.from_square)
            if piece is None:
                logger.error(f"No piece at square {chess.square_name(move.from_square)}")
                board.push(move)
                return "Geçersiz hamle"
                
            from_square = chess.square_name(move.from_square)
            to_square = chess.square_name(move.to_square)
            is_capture = board.is_capture(move)
            
            # Make the move
            board.push(move)
            
            # Build Turkish-specific commentary with proper suffixes
            comment = self.name(piece.piece_type)
            comment += f" {from_square}"
            
            if is_capture:
                comment += " alır"
                comment += f" {to_square}'{self.capture_suffix(to_square)}"
            else:
                comment += f"'{self.from_suffix(from_square)} {to_square}'{self.to_suffix(to_square)}"
            
            # Handle promotion
            if move.promotion:
                comment += " "
                if move.promotion == chess.KNIGHT:
                    comment += "ata"
                elif move.promotion == chess.BISHOP:
                    comment += "file"
                elif move.promotion == chess.ROOK:
                    comment += "kaleye"
                elif move.promotion == chess.QUEEN:
                    comment += "vezire"
                comment += " terfi"
                
            comment += check_status
            return comment
            
        except Exception as e:
            logger.error(f"Error generating Turkish commentary: {e}")
            return "Hamle yapıldı"


class Italian(LanguageBase):
    """Italian language for chess commentary."""
    
    def __init__(self):
        super().__init__()
        self.game_started = "Gioco iniziato"
        self.move_failed = "Registrazione spostamento non riuscita. Per favore rifai la tua mossa."
        self.game_over = "Partita finita"
        self.white_wins = "Il bianco vince"
        self.black_wins = "Il nero vince"
        self.draw = "Patta"
        self.game_over_resignation_or_draw = "Partita terminata per abbandono o patta"
        
        # Check and checkmate
        self.check = "scacco"
        self.checkmate = "scacco matto"
        
        # Castling
        self.castling_kingside = "arrocco corto"
        self.castling_queenside = "arrocco lungo"
    
    def name(self, piece_type: int) -> str:
        """Get Italian name for chess piece."""
        piece_names = {
            chess.PAWN: "pedone",
            chess.KNIGHT: "cavallo",
            chess.BISHOP: "alfiere",
            chess.ROOK: "torre",
            chess.QUEEN: "regina",
            chess.KING: "re"
        }
        
        if piece_type not in piece_names:
            logger.error(f"Invalid piece type: {piece_type}")
            return "sconosciuto"
            
        return piece_names[piece_type]
    
    def prefix_name(self, piece_type: int) -> str:
        """Get Italian prefixed name (with article) for chess piece."""
        try:
            prefixed_names = {
                chess.PAWN: "il pedone",
                chess.KNIGHT: "il cavallo",
                chess.BISHOP: "l'alfiere",
                chess.ROOK: "la torre",
                chess.QUEEN: "la regina",
                chess.KING: "il re"
            }
            
            if piece_type not in prefixed_names:
                logger.error(f"Invalid piece type for prefix: {piece_type}")
                return "il pezzo"
                
            return prefixed_names[piece_type]
        except Exception as e:
            logger.error(f"Error getting Italian prefix name: {e}")
            return "il pezzo"
    
    def comment(self, board: chess.Board, move: chess.Move) -> str:
        """Generate Italian commentary for a chess move."""
        try:
            # Check for special states
            check_status = ""
            if board.is_checkmate():
                check_status = f" {self.checkmate}"
            elif board.is_check():
                check_status = f" {self.check}"
                
            # Revert the board to check castling and movement
            board.pop()
            
            # Check for castling moves
            if board.is_kingside_castling(move):
                board.push(move)
                return f"{self.castling_kingside}{check_status}"
                
            if board.is_queenside_castling(move):
                board.push(move)
                return f"{self.castling_queenside}{check_status}"
            
            # Get piece information and squares
            piece = board.piece_at(move.from_square)
            if piece is None:
                logger.error(f"No piece at square {chess.square_name(move.from_square)}")
                board.push(move)
                return "Mossa non valida"
                
            from_square = chess.square_name(move.from_square)
            to_square = chess.square_name(move.to_square)
            is_capture = board.is_capture(move)
            
            # Check for promotion
            promotion_text = ""
            if move.promotion:
                promotion_text = f" promuove a {self.name(move.promotion)}"
            
            # Make the move
            board.push(move)
            
            # Build Italian-specific commentary
            if is_capture:
                comment = f"{self.prefix_name(piece.piece_type)} {from_square} cattura {to_square}"
            else:
                comment = f"{self.name(piece.piece_type)} da {from_square} a {to_square}"
            
            comment += f"{promotion_text}{check_status}"
            return comment
            
        except Exception as e:
            logger.error(f"Error generating Italian commentary: {e}")
            return "Mossa effettuata"


class French(LanguageBase):
    """French language for chess commentary."""
    
    def __init__(self):
        super().__init__()
        self.game_started = "Partie démarrée"
        self.move_failed = "La reconnaissance a échoué. Veuillez réessayer."
        self.game_over = "Partie terminée"
        self.white_wins = "Les blancs gagnent"
        self.black_wins = "Les noirs gagnent"
        self.draw = "Nulle"
        self.game_over_resignation_or_draw = "Partie terminée par abandon ou nulle"
        
        # Piece movement modifiers
        self.captures = "prend"
        self.moves_to = "vers"
        self.promotion_text = "promu en"
        
        # Check and checkmate
        self.check = "échec"
        self.checkmate = "échec et mat"
        
        # Castling
        self.castling_kingside = "petit roc"
        self.castling_queenside = "grand roc"
    
    def name(self, piece_type: int) -> str:
        """Get French name for chess piece."""
        piece_names = {
            chess.PAWN: "pion",
            chess.KNIGHT: "cavalier",
            chess.BISHOP: "fou",
            chess.ROOK: "tour",
            chess.QUEEN: "reine",
            chess.KING: "roi"
        }
        
        if piece_type not in piece_names:
            logger.error(f"Invalid piece type: {piece_type}")
            return "pièce inconnue"
            
        return piece_names[piece_type]


# Language registry to easily get language by name
LANGUAGES: Dict[str, Type[LanguageBase]] = {
    "english": English,
    "german": German,
    "russian": Russian,
    "turkish": Turkish,
    "italian": Italian,
    "french": French
}

def get_language(language_name: str) -> LanguageBase:
    """
    Get a language instance by name.
    
    Args:
        language_name: Name of the language (case-insensitive)
        
    Returns:
        Language instance
        
    Raises:
        ValueError: If language is not supported
    """
    language_key = language_name.lower()
    if language_key in LANGUAGES:
        return LANGUAGES[language_key]()
    else:
        logger.warning(f"Unsupported language: {language_name}, falling back to English")
        return English()


# Test function
def test_languages():
    """Test the language functionality with a sample move."""
    board = chess.Board()
    
    # Standard pawn move
    e2e4 = chess.Move.from_uci("e2e4")
    board.push(e2e4)
    
    # Test all languages
    for lang_name, lang_class in LANGUAGES.items():
        lang = lang_class()
        comment = lang.comment(board, e2e4)
        print(f"{lang_name.capitalize()}: {comment}")
    
    # Reset board and test capture
    board = chess.Board()
    board.push(chess.Move.from_uci("e2e4"))
    board.push(chess.Move.from_uci("d7d5"))
    capture = chess.Move.from_uci("e4d5")
    board.push(capture)
    
    # Test castling
    board = chess.Board()
    board.push(chess.Move.from_uci("e2e4"))
    board.push(chess.Move.from_uci("e7e5"))
    board.push(chess.Move.from_uci("g1f3"))
    board.push(chess.Move.from_uci("b8c6"))
    board.push(chess.Move.from_uci("f1c4"))
    board.push(chess.Move.from_uci("f8c5"))
    castling = chess.Move.from_uci("e1g1")  # White kingside castling
    
    english = English()
    print(f"Castling: {english.comment(board, castling)}")
    
    # Test with promotion
    board = chess.Board("rnbqkbnr/pPpppppp/8/8/8/8/1PPPPPPP/RNBQKBNR w KQkq - 0 1")  # White pawn on b7
    promotion = chess.Move.from_uci("b7b8q")  # Promote to queen
    board.push(promotion)
    
    for lang_name, lang_class in LANGUAGES.items():
        lang = lang_class()
        comment = lang.comment(board, promotion)
        print(f"{lang_name.capitalize()} promotion: {comment}")


if __name__ == "__main__":
    test_languages()
