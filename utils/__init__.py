"""utils — libreria importabile per notebook e script.

Re-export delle classi/funzioni più usate per consentire import compatti:

    from utils import KGScorer

equivalente a `from utils.kg import KGScorer` ma più breve.
Per gli helper di text processing si continua a usare l'import esplicito
`from utils.text_processing import segment_article` (sono funzioni internal
di un singolo notebook, non API generale).
"""

from utils.kg import KGScorer

__all__ = ["KGScorer"]