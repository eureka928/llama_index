import pytest
from importlib.util import find_spec
from unittest.mock import MagicMock, patch
from llama_index.core.storage.kvstore.types import BaseKVStore
from llama_index.storage.kvstore.postgres import PostgresKVStore

no_packages = (
    find_spec("psycopg2") is None
    or find_spec("sqlalchemy") is None
    or find_spec("asyncpg") is None
)


def test_class():
    names_of_base_classes = [b.__name__ for b in PostgresKVStore.__mro__]
    assert BaseKVStore.__name__ in names_of_base_classes


@pytest.mark.skipif(
    no_packages, reason="asyncpg, pscopg2-binary and sqlalchemy not installed"
)
def test_initialization():
    errors = []
    try:
        pgstore1 = PostgresKVStore(table_name="mytable")
        errors.append(0)
    except ValueError:
        errors.append(1)
    try:
        pgstore2 = PostgresKVStore(
            table_name="mytable", connection_string="connection_string"
        )
        errors.append(0)
    except ValueError:
        errors.append(1)
    try:
        pgstore3 = PostgresKVStore(
            table_name="mytable", async_connection_string="async_connection_string"
        )
        errors.append(0)
    except ValueError:
        errors.append(1)
    try:
        pgstore4 = PostgresKVStore(
            table_name="mytable",
            connection_string="connection_string",
            async_connection_string="async_connection_string",
        )
        errors.append(0)
    except ValueError:
        errors.append(1)
    assert sum(errors) == 3
    assert pgstore4._engine is None
    assert pgstore4._async_engine is None


@pytest.mark.skipif(
    no_packages, reason="asyncpg, pscopg2-binary and sqlalchemy not installed"
)
def test_schema_creation_uses_inspect_when_schema_does_not_exist():
    import sqlalchemy

    mock_engine = MagicMock()
    mock_async_engine = MagicMock()

    mock_session_instance = MagicMock()
    mock_session_instance.connection.return_value = MagicMock()

    mock_session_ctx = MagicMock()
    mock_session_ctx.__enter__.return_value = mock_session_instance
    mock_session_ctx.__exit__.return_value = None

    mock_begin_ctx = MagicMock()
    mock_begin_ctx.__enter__.return_value = MagicMock()
    mock_begin_ctx.__exit__.return_value = None
    mock_session_instance.begin.return_value = mock_begin_ctx

    mock_session_factory = MagicMock(return_value=mock_session_ctx)

    mock_inspector = MagicMock()
    mock_inspector.get_schema_names.return_value = []

    with (
        patch.object(sqlalchemy, "create_engine", return_value=mock_engine),
        patch.object(
            sqlalchemy.ext.asyncio,
            "create_async_engine",
            return_value=mock_async_engine,
        ),
        patch("sqlalchemy.orm.sessionmaker", return_value=mock_session_factory),
        patch(
            "llama_index.storage.kvstore.postgres.base.inspect",
            return_value=mock_inspector,
        ),
    ):
        pgstore = PostgresKVStore(
            table_name="test_table",
            connection_string="postgresql://user:pass@localhost/db",
            async_connection_string="postgresql+asyncpg://user:pass@localhost/db",
            schema_name="test_schema",
            perform_setup=False,
        )

        pgstore._connect()
        pgstore._create_schema_if_not_exists()

        from llama_index.storage.kvstore.postgres.base import inspect

        inspect.assert_called_once_with(mock_session_instance.connection())
        mock_inspector.get_schema_names.assert_called_once()

        execute_calls = mock_session_instance.execute.call_args_list
        assert len(execute_calls) == 1

        from sqlalchemy.schema import CreateSchema

        executed_statement = execute_calls[0][0][0]
        assert isinstance(executed_statement, CreateSchema)
        assert executed_statement.element == "test_schema"


@pytest.mark.skipif(
    no_packages, reason="asyncpg, pscopg2-binary and sqlalchemy not installed"
)
def test_schema_creation_uses_inspect_when_schema_exists():
    import sqlalchemy

    mock_engine = MagicMock()
    mock_async_engine = MagicMock()

    mock_session_instance = MagicMock()
    mock_session_instance.connection.return_value = MagicMock()

    mock_session_ctx = MagicMock()
    mock_session_ctx.__enter__.return_value = mock_session_instance
    mock_session_ctx.__exit__.return_value = None

    mock_begin_ctx = MagicMock()
    mock_begin_ctx.__enter__.return_value = MagicMock()
    mock_begin_ctx.__exit__.return_value = None
    mock_session_instance.begin.return_value = mock_begin_ctx

    mock_session_factory = MagicMock(return_value=mock_session_ctx)

    mock_inspector = MagicMock()
    mock_inspector.get_schema_names.return_value = ["test_schema"]

    with (
        patch.object(sqlalchemy, "create_engine", return_value=mock_engine),
        patch.object(
            sqlalchemy.ext.asyncio,
            "create_async_engine",
            return_value=mock_async_engine,
        ),
        patch("sqlalchemy.orm.sessionmaker", return_value=mock_session_factory),
        patch(
            "llama_index.storage.kvstore.postgres.base.inspect",
            return_value=mock_inspector,
        ),
    ):
        pgstore = PostgresKVStore(
            table_name="test_table",
            connection_string="postgresql://user:pass@localhost/db",
            async_connection_string="postgresql+asyncpg://user:pass@localhost/db",
            schema_name="test_schema",
            perform_setup=False,
        )

        pgstore._connect()
        pgstore._create_schema_if_not_exists()

        from llama_index.storage.kvstore.postgres.base import inspect

        inspect.assert_called_once_with(mock_session_instance.connection())
        mock_inspector.get_schema_names.assert_called_once()

        mock_session_instance.execute.assert_not_called()


@pytest.mark.skipif(
    no_packages, reason="asyncpg, pscopg2-binary and sqlalchemy not installed"
)
def test_put_all_uses_safe_insert():
    import sqlalchemy

    mock_engine = MagicMock()
    mock_async_engine = MagicMock()

    mock_session_instance = MagicMock()
    mock_session_ctx = MagicMock()
    mock_session_ctx.__enter__.return_value = mock_session_instance
    mock_session_ctx.__exit__.return_value = None

    mock_session_factory = MagicMock(return_value=mock_session_ctx)

    with (
        patch.object(sqlalchemy, "create_engine", return_value=mock_engine),
        patch.object(
            sqlalchemy.ext.asyncio,
            "create_async_engine",
            return_value=mock_async_engine,
        ),
        patch("sqlalchemy.orm.sessionmaker", return_value=mock_session_factory),
    ):
        pgstore = PostgresKVStore(
            table_name="test_table",
            connection_string="postgresql://user:pass@localhost/db",
            async_connection_string="postgresql+asyncpg://user:pass@localhost/db",
            schema_name="test_schema",
            perform_setup=False,
        )
        pgstore._connect()
        pgstore._is_initialized = True

        test_data = [("key1", {"value": "data1"}), ("key2", {"value": "data2"})]
        pgstore.put_all(test_data)

        execute_calls = mock_session_instance.execute.call_args_list
        assert len(execute_calls) >= 1

        executed_statement = execute_calls[-1][0][0]
        assert hasattr(executed_statement, "compile")


@pytest.mark.skipif(
    no_packages, reason="asyncpg, pscopg2-binary and sqlalchemy not installed"
)
@pytest.mark.asyncio
async def test_aput_all_uses_safe_insert():
    import sqlalchemy
    from unittest.mock import AsyncMock

    mock_engine = MagicMock()
    mock_async_engine = MagicMock()

    mock_session_instance = AsyncMock()
    mock_begin_ctx_manager = MagicMock()
    mock_begin_ctx_manager.__aenter__ = AsyncMock()
    mock_begin_ctx_manager.__aexit__ = AsyncMock()
    mock_session_instance.begin.return_value = mock_begin_ctx_manager

    mock_session_ctx_manager = MagicMock()
    mock_session_ctx_manager.__aenter__ = AsyncMock(return_value=mock_session_instance)
    mock_session_ctx_manager.__aexit__ = AsyncMock(return_value=None)

    mock_async_session_factory = MagicMock(return_value=mock_session_ctx_manager)

    with (
        patch.object(sqlalchemy, "create_engine", return_value=mock_engine),
        patch.object(
            sqlalchemy.ext.asyncio,
            "create_async_engine",
            return_value=mock_async_engine,
        ),
        patch("sqlalchemy.orm.sessionmaker", return_value=mock_async_session_factory),
    ):
        pgstore = PostgresKVStore(
            table_name="test_table",
            connection_string="postgresql://user:pass@localhost/db",
            async_connection_string="postgresql+asyncpg://user:pass@localhost/db",
            schema_name="test_schema",
            perform_setup=False,
        )
        pgstore._connect()
        pgstore._is_initialized = True

        mock_session_instance.execute = AsyncMock()

        test_data = [("key1", {"value": "data1"}), ("key2", {"value": "data2"})]
        await pgstore.aput_all(test_data)

        execute_calls = mock_session_instance.execute.call_args_list
        assert len(execute_calls) >= 1

        executed_statement = execute_calls[-1][0][0]
        assert hasattr(executed_statement, "compile")


@pytest.mark.skipif(
    no_packages, reason="asyncpg, pscopg2-binary and sqlalchemy not installed"
)
def test_schema_name_with_special_characters():
    import sqlalchemy

    mock_engine = MagicMock()
    mock_async_engine = MagicMock()

    mock_session_instance = MagicMock()
    mock_session_ctx = MagicMock()
    mock_session_ctx.__enter__.return_value = mock_session_instance
    mock_session_ctx.__exit__.return_value = None

    mock_begin_ctx = MagicMock()
    mock_begin_ctx.__enter__.return_value = MagicMock()
    mock_begin_ctx.__exit__.return_value = None
    mock_session_instance.begin.return_value = mock_begin_ctx

    mock_session_factory = MagicMock(return_value=mock_session_ctx)

    mock_inspector = MagicMock()
    mock_inspector.get_schema_names.return_value = []

    with (
        patch.object(sqlalchemy, "create_engine", return_value=mock_engine),
        patch.object(
            sqlalchemy.ext.asyncio,
            "create_async_engine",
            return_value=mock_async_engine,
        ),
        patch("sqlalchemy.orm.sessionmaker", return_value=mock_session_factory),
        patch(
            "llama_index.storage.kvstore.postgres.base.inspect",
            return_value=mock_inspector,
        ),
    ):
        special_schema = "test'schema"
        pgstore = PostgresKVStore(
            table_name="test_table",
            connection_string="postgresql://user:pass@localhost/db",
            async_connection_string="postgresql+asyncpg://user:pass@localhost/db",
            schema_name=special_schema,
            perform_setup=False,
        )

        pgstore._connect()
        pgstore._create_schema_if_not_exists()

        execute_calls = mock_session_instance.execute.call_args_list
        assert len(execute_calls) == 1

        from sqlalchemy.schema import CreateSchema

        executed_statement = execute_calls[0][0][0]
        assert isinstance(executed_statement, CreateSchema)


@pytest.mark.skipif(
    no_packages, reason="asyncpg, pscopg2-binary and sqlalchemy not installed"
)
def test_create_engine_kwargs_passed_to_engines():
    """Verify create_engine_kwargs are forwarded to both sync and async engines."""
    import sqlalchemy

    mock_engine = MagicMock()
    mock_async_engine = MagicMock()

    engine_kwargs = {
        "pool_size": 10,
        "max_overflow": 20,
        "pool_pre_ping": True,
    }

    with (
        patch.object(
            sqlalchemy, "create_engine", return_value=mock_engine
        ) as mock_create_engine,
        patch.object(
            sqlalchemy.ext.asyncio,
            "create_async_engine",
            return_value=mock_async_engine,
        ) as mock_create_async_engine,
        patch("sqlalchemy.orm.sessionmaker"),
    ):
        pgstore = PostgresKVStore(
            table_name="test_table",
            connection_string="postgresql://user:pass@localhost/db",
            async_connection_string="postgresql+asyncpg://user:pass@localhost/db",
            perform_setup=False,
            create_engine_kwargs=engine_kwargs,
        )
        pgstore._connect()

        mock_create_engine.assert_called_once_with(
            "postgresql://user:pass@localhost/db",
            echo=False,
            pool_size=10,
            max_overflow=20,
            pool_pre_ping=True,
        )
        mock_create_async_engine.assert_called_once_with(
            "postgresql+asyncpg://user:pass@localhost/db",
            pool_size=10,
            max_overflow=20,
            pool_pre_ping=True,
        )


@pytest.mark.skipif(
    no_packages, reason="asyncpg, pscopg2-binary and sqlalchemy not installed"
)
def test_create_engine_kwargs_default_empty():
    """Verify create_engine_kwargs defaults to empty dict when not provided."""
    pgstore = PostgresKVStore(
        table_name="test_table",
        connection_string="postgresql://user:pass@localhost/db",
        async_connection_string="postgresql+asyncpg://user:pass@localhost/db",
        perform_setup=False,
    )
    assert pgstore.create_engine_kwargs == {}


@pytest.mark.skipif(
    no_packages, reason="asyncpg, pscopg2-binary and sqlalchemy not installed"
)
def test_from_uri_forwards_create_engine_kwargs():
    """Verify from_uri forwards create_engine_kwargs to constructor."""
    engine_kwargs = {"pool_size": 5, "connect_args": {"timeout": 60}}

    with patch.object(PostgresKVStore, "__init__", return_value=None) as mock_init:
        PostgresKVStore.from_uri(
            uri="postgresql://user:pass@localhost:5432/db",
            table_name="test_table",
            create_engine_kwargs=engine_kwargs,
        )

        # from_uri calls from_params which calls __init__
        mock_init.assert_called_once()
        call_kwargs = mock_init.call_args
        assert call_kwargs.kwargs.get("create_engine_kwargs") == engine_kwargs


@pytest.mark.skipif(
    no_packages, reason="asyncpg, pscopg2-binary and sqlalchemy not installed"
)
def test_from_params_forwards_create_engine_kwargs():
    """Verify from_params forwards create_engine_kwargs to constructor."""
    engine_kwargs = {"pool_size": 5}

    with patch.object(PostgresKVStore, "__init__", return_value=None) as mock_init:
        PostgresKVStore.from_params(
            host="localhost",
            port="5432",
            database="db",
            user="user",
            password="pass",
            table_name="test_table",
            create_engine_kwargs=engine_kwargs,
        )

        mock_init.assert_called_once()
        call_kwargs = mock_init.call_args
        assert call_kwargs.kwargs.get("create_engine_kwargs") == engine_kwargs
